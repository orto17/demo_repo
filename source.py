"""Training Operator Base Executor"""

import logging
import os
import time
from pathlib import PurePath
from typing import Any, Callable, Union

import yaml
from kubeflow.training.api import tf_job_client
from kubernetes import client
from tfx.dsl.io import fileio
from tfx.types import Artifact, artifact_utils, standard_component_specs
from tfx.utils import io_utils, path_utils

from blueyonder.exec.ml.kubernetes import Pod
from blueyonder.exec.ml.ops import CurrentPodContainerImageNotFoundException
from blueyonder.exec.ml.ops.tfx.component.executor import EMLBaseExecutor
from blueyonder.exec.ml.utils import kubeflow, properties
from blueyonder.exec.ml.utils.configuration_loader import ConfigurationLoader
from blueyonder.exec.ml.utils.multi_tenancy import resolve_tenant_client_file

_LOGGER = logging.getLogger(__name__)


class Executor(EMLBaseExecutor):
    """
    Custom TFX component to launch distributed training jobs
    """

    def do_internal(
        self,
        input_dict: dict[str, list[Artifact]],
        output_dict: dict[str, list[Artifact]],
        exec_properties: dict[str, Any],
    ):
        """Execution starts here."""
        self._output_dict = output_dict
        self._exec_properties = exec_properties
        self._k8s_apps_client = None
        self._k8s_core_client = None

        self._configuration: ConfigurationLoader = ConfigurationLoader(
            configuration=self.get_configuration(), module=exec_properties["module"]
        )
        self._experiment_yaml_path: PurePath = resolve_tenant_client_file(
            file_name=exec_properties.get("experiment_yaml")
        )
        job_yaml_file = exec_properties.get("job_yaml")
        if job_yaml_file:
            self._yaml_name = job_yaml_file

        self._job_name: str = self._get_job_name()
        self._namespace: str = kubeflow.get_namespace()

        self._tf_data_service_enabled = exec_properties.get("tf_data_service_dispatcher") is True
        if self._tf_data_service_enabled:
            self._set_tfdata_deployment_names()

        _LOGGER.info(
            f"Populating model run directory if resuming from previous run: {exec_properties.get('resume_from')}"
        )
        self._maybe_populate_model_run_dir(resume_from=exec_properties.get("resume_from"))
        _LOGGER.info("Persisting or reusing experiment YAML file.")
        self._persist_or_reuse_experiment_yaml()
        self._set_job_type()

        args: list[str] = self._prepare_args(input_dict, output_dict, exec_properties["module"])
        job_body: dict = self._prepare_job_yaml(exec_properties=exec_properties, args=args)
        self._job_body: dict = self._modify_job_body(job_body)

        if self._tf_data_service_enabled:
            self._deploy_tfdata_service()
            self._wait_for_tfdata_service_ready()

        _LOGGER.info(f"Job yaml:\n\n{yaml.dump(self._job_body)}")

        self._client = tf_job_client.TFJobClient()
        self._n_fails: int = 0

        if not self._is_job_running():
            _LOGGER.info(f"No running job found for {self._job_name}, launching new job.")
            self._launch_job()
        else:
            _LOGGER.info(f"Job {self._job_name} is already running.")

        try:
            self._track_job()

            if self._client.is_job_succeeded(
                name=self._job_name,
                namespace=self._namespace,
                kind=self._kind,
            ):
                _LOGGER.info(f"Job {self._job_name} completed successfully")
            else:
                raise ValueError(f"Job {self._job_name} failed")
        finally:
            if self._tf_data_service_enabled:
                self._cleanup_tfdata_service()

    def _update_experiment_yaml_dispatcher_address(self, write_yaml_path: str):
        """Update the dispatcher address in the experiment YAML file on disk on the fly"""
        yaml_path = str(self._experiment_yaml_path)
        with fileio.open(yaml_path, "r") as f:
            experiment_config = yaml.safe_load(f)

        if "task" in experiment_config and isinstance(experiment_config["task"], dict):
            experiment_config["task"]["tf_data_service_dispatcher"] = self._dispatcher_address
            _LOGGER.info(f"Updated tf_data_service_dispatcher to persistent service: {self._dispatcher_address}")
        else:
            _LOGGER.warning("Could not find 'task' section in experiment YAML to update dispatcher address.")

        with fileio.open(write_yaml_path, "w") as f:
            yaml.safe_dump(experiment_config, f)
            _LOGGER.info(f"Experiment YAML written to: {write_yaml_path}")

    def _set_job_type(self):
        """Set job type in child classes"""
        self._yaml_name = ""
        self._spec_name = ""
        self._restart_policy = {}
        self._modify_job_body = lambda x: x
        self._fail_tolerance = -1
        self._kind = ""
        self._main_replica = ""
        raise NotImplementedError

    def _maybe_populate_model_run_dir(self, resume_from: Union[str, None]):
        """Copy over logs and checkpoints from previous model run when resume_from is set"""
        if resume_from:
            model_run_dir: str = self._get_model_run_dir()
            resume_src_dir: str = self._get_model_run_dir(model_uri=resume_from)
            _LOGGER.info(f"Copying logs and checkpoints from {resume_src_dir} to {model_run_dir}")
            io_utils.copy_dir(src=resume_src_dir, dst=model_run_dir)

    def _persist_or_reuse_experiment_yaml(self):
        """Persist experiment yaml if not already persisted.
        Note: Training will not reuse experiment yaml from a different run (eg: when resume_from is set).
        """

        model_dir: str = self._get_model_run_dir()
        yaml_path = os.path.join(model_dir, self._experiment_yaml_path.name)
        if not fileio.exists(yaml_path):
            _LOGGER.info(f"Persisting experiment YAML to {yaml_path}")
            if self._tf_data_service_enabled:
                _LOGGER.info("\n\nRewriting Experiment YAML with TF Data Dispatcher Address\n\n")
                self._update_experiment_yaml_dispatcher_address(write_yaml_path=yaml_path)
            else:
                io_utils.copy_file(
                    src=str(self._experiment_yaml_path),
                    dst=os.path.join(model_dir),
                    overwrite=True,
                )
        else:
            _LOGGER.info(f"Experiment YAML already exists at {yaml_path}, skipping copy.")

    def _get_model_run_dir(self, model_uri=None) -> str:
        """Generate model run directory uri from model uri.
        Model run directory is where training progress is saved.
        Takes an optional model_uri argument, to reuse the utility for different model uris.
        """
        model_uri: str = model_uri or artifact_utils.get_single_uri(
            artifact_list=self._output_dict[standard_component_specs.MODEL_KEY]
        )
        uri: list[str] = model_uri.split("/")
        uri[-2] = standard_component_specs.MODEL_RUN_KEY
        uri: str = "/".join(uri)
        return uri

    def _prepare_args(
        self, input_dict: dict[str, list[Artifact]], output_dict: dict[str, list[Artifact]], module: str
    ) -> list[str]:
        """Prepare args for calling training entrypoint."""
        args = []

        file_pattern: Callable[[list[Artifact], str], str] = lambda artifacts, split: artifact_utils.get_split_uris(
            artifact_list=artifacts, split=split
        )[0]
        train_files: str = io_utils.all_files_pattern(
            file_pattern=file_pattern(
                artifacts=input_dict[standard_component_specs.EXAMPLES_KEY],
                split="train",
            )
        )
        args.append(f"--train_files={train_files}")
        eval_files: str = io_utils.all_files_pattern(
            file_pattern=file_pattern(
                artifacts=input_dict[standard_component_specs.EXAMPLES_KEY],
                split="eval",
            )
        )
        args.append(f"--eval_files={eval_files}")

        if input_dict.get(standard_component_specs.STATISTICS_KEY) is not None:
            statistics_path: str = (
                artifact_utils.get_single_uri(artifact_list=input_dict[standard_component_specs.STATISTICS_KEY])
                + "/Split-train"
            )
            args.append(f"--statistics_path={statistics_path}")

        if input_dict.get(standard_component_specs.SCHEMA_KEY) is not None:
            schema_path: str = artifact_utils.get_single_uri(
                artifact_list=input_dict[standard_component_specs.SCHEMA_KEY]
            )
            args.append(f"--schema_path={schema_path}")

        if input_dict.get(standard_component_specs.TRANSFORM_GRAPH_KEY) is not None:
            transform_dir: str = artifact_utils.get_single_uri(
                artifact_list=input_dict[standard_component_specs.TRANSFORM_GRAPH_KEY]
            )
            args.append(f"--transform_dir={transform_dir}")

        serving_model_dir: str = path_utils.serving_model_dir(
            output_uri=artifact_utils.get_single_uri(artifact_list=output_dict[standard_component_specs.MODEL_KEY])
        )
        args.append(f"--serving_model_dir={serving_model_dir}")

        model_dir: str = self._get_model_run_dir()
        args.append(f"--model_dir={model_dir}")

        args.append(f"--module={module}")
        args.append(f"--experiment_yaml={os.path.join(model_dir, self._experiment_yaml_path.name)}")
        args.append(f"--json_config={self._configuration.serialize()}")

        if self._tf_data_service_enabled:
            args.append(f"--tf_data_service_dispatcher={self._dispatcher_address}")
            _LOGGER.info(f"Added tf.data service dispatcher argument: {self._dispatcher_address}")

        return args

    def _set_tfdata_deployment_names(self):
        """Sets unique names for the tf.data deployment resources for this run."""
        self._tfdata_dispatcher_name = f"{self._job_name}-dispatcher"
        self._tfdata_worker_name = f"{self._job_name}-worker"
        self._tfdata_dispatcher_service_name = "tfdata-dispatcher"
        self._tfdata_worker_service_name = "tfdata-worker"
        self._dispatcher_address = (
            f"grpc://{self._tfdata_dispatcher_service_name}.{self._namespace}.svc.cluster.local:5000"
        )
        _LOGGER.info(f"Using ephemeral deployment names: {self._tfdata_dispatcher_name}, {self._tfdata_worker_name}")
        _LOGGER.info(
            f"Using persistent service names: {self._tfdata_dispatcher_service_name}, {self._tfdata_worker_service_name}"
        )
        _LOGGER.info(f"Set dispatcher address to: {self._dispatcher_address}")

    def _deploy_tfdata_service(self):
        """Deploy tf.data service dispatcher following project conventions."""
        try:
            tfdata_yaml_body: dict = self._get_tfdata_yaml()
            _LOGGER.info(f"tf.data service dispatcher YAML:\n\n{yaml.dump(tfdata_yaml_body)}")

            self._create_tfdata_resources(tfdata_yaml_body)
            _LOGGER.info("tf.data service dispatcher deployed successfully")

        except Exception as e:
            _LOGGER.warning(f"Failed to deploy tf.data service: {e}")
            _LOGGER.info("Training will continue without tf.data service sharding")

    def _cleanup_tfdata_service(self):
        """Deletes the tf.data service deployments."""
        _LOGGER.info("Cleaning up tf.data service resources...")
        apps_client = Pod.get_k8s_apps_client(self._k8s_apps_client)

        deployments_to_delete = [self._tfdata_dispatcher_name, self._tfdata_worker_name]

        for deployment_name in deployments_to_delete:
            try:
                _LOGGER.info(f"Deleting Deployment: {deployment_name} in namespace {self._namespace}")
                apps_client.delete_namespaced_deployment(name=deployment_name, namespace=self._namespace)
                _LOGGER.info(f"Successfully deleted deployment: {deployment_name}")
            except client.rest.ApiException as e:
                if e.status == 404:
                    _LOGGER.warning(f"Deployment '{deployment_name}' not found, may have already been deleted.")
                else:
          #          _LOGGER.error(f"Error deleting deployment '{deployment_name}': {e}")
            except Exception as e:
          #      _LOGGER.error(f"An unexpected error occurred while deleting deployment '{deployment_name}': {e}")

        # Services are preserved for reuse by future training runs
        _LOGGER.info(
            f"Persistent services preserved: {self._tfdata_dispatcher_service_name}, {self._tfdata_worker_service_name}"
        )

    def _wait_for_tfdata_service_ready(self, timeout_seconds: int = 600):
        """
        Waits for the tf.data service dispatcher and worker deployments to become ready.

        Polls the status of the deployments until the number of ready replicas
        matches the desired number of replicas.

        Args:
            timeout_seconds: The maximum time in seconds to wait for the services.
        """
        apps_client = Pod.get_k8s_apps_client(self._k8s_apps_client)
        services_to_check = [self._tfdata_dispatcher_name, self._tfdata_worker_name]
        start_time = time.time()

        _LOGGER.info(f"Waiting for tf.data service deployments in namespace '{self._namespace}' to become ready...")

        while time.time() - start_time < timeout_seconds:
            all_services_ready = True
            try:
                for service_name in services_to_check:
                    try:
                        deployment = apps_client.read_namespaced_deployment(
                            name=service_name, namespace=self._namespace
                        )
                        desired_replicas = deployment.spec.replicas
                        ready_replicas = deployment.status.ready_replicas or 0

                        if ready_replicas < desired_replicas:
                            _LOGGER.info(
                                f"Deployment '{service_name}' is not ready yet ({ready_replicas}/{desired_replicas} replicas ready)."
                            )
                            all_services_ready = False
                            break
                        else:
                            _LOGGER.info(
                                f"Deployment '{service_name}' is ready ({ready_replicas}/{desired_replicas} replicas ready)."
                            )

                    except client.rest.ApiException as e:
                        if e.status == 404:
                            _LOGGER.info(f"Deployment '{service_name}' not found yet, waiting for creation...")
                            all_services_ready = False
                            break
                        else:
                    #        _LOGGER.warning(f"API error while checking deployment '{service_name}': {e}")
                            all_services_ready = False
                            break

                if all_services_ready:
                    _LOGGER.info("All tf.data service deployments are ready.")
                    return

            except Exception as e:
             #   _LOGGER.warning(f"Unexpected error while checking tf.data service status: {e}. Retrying...")
                all_services_ready = False

            time.sleep(30)

        raise TimeoutError(f"Timed out after {timeout_seconds} seconds waiting for tf.data service to become ready.")

    def _get_tfdata_yaml(self) -> dict:
        """Fetch and update tf.data dispatcher YAML for resource deployment"""
        tfdata_file_name: str = "tfdata-dispatcher.yaml"
        try:
            tfdata_file: str = properties.get_file_path(tfdata_file_name)
        except ValueError:
            raise ValueError(f"tf.data service dispatcher YAML not found: {tfdata_file_name}")

        with fileio.open(tfdata_file, "r") as f:
            tfdata_yaml_content = f.read()

        # Parse YAML documents (deployment and service) - handle multiple documents
        tfdata_resources = list(yaml.safe_load_all(tfdata_yaml_content))

        for resource in tfdata_resources:
            if resource:
                # Update namespace to match training job namespace
                resource["metadata"]["namespace"] = self._namespace
                kind = resource.get("kind")

                if kind in ["Deployment", "Service"]:
                    is_dispatcher = "dispatcher" in resource["metadata"]["name"]
                    # Update container image for deployment using same method as training job
                    if kind == "Deployment":
                        # Use ephemeral deployment names (unique per pipeline run)
                        new_name = self._tfdata_dispatcher_name if is_dispatcher else self._tfdata_worker_name
                        resource["metadata"]["name"] = new_name

                        # Fix deployment selector and pod labels to match
                        deployment_labels = {"app": new_name}
                        resource["spec"]["selector"]["matchLabels"] = deployment_labels
                        resource["spec"]["template"]["metadata"]["labels"] = deployment_labels.copy()

                        # Add tf-data-service label for service selection
                        resource["spec"]["template"]["metadata"]["labels"]["tf-data-service"] = (
                            "dispatcher" if is_dispatcher else "worker"
                        )

                        # Set container image following project conventions
                        self._set_tfdata_container_image(tfdata_yaml=resource)

                        if not is_dispatcher:
                            container = resource["spec"]["template"]["spec"]["containers"][0]
                            original_command_module = " ".join(container["command"][2:])
                            original_args = container["args"]

                            # Find and update the --dispatcher_address argument in the original args
                            dispatcher_fqdn = (
                                f"{self._tfdata_dispatcher_service_name}.{self._namespace}.svc.cluster.local"
                            )
                            for i, arg in enumerate(original_args):
                                if arg.startswith("--dispatcher_address"):
                                    original_args[i] = f"--dispatcher_address={dispatcher_fqdn}:5000"
                                    _LOGGER.info(
                                        f"Updated tf.data worker dispatcher address to constant service: {dispatcher_fqdn}:5000"
                                    )
                                    break

                            # Construct the full python command with its arguments
                            python_command = f"python -m {original_command_module} {' '.join(original_args)}"

                            # Wrap the full python command in a shell script that waits for DNS.
                            wait_command = [
                                "/bin/sh",
                                "-c",
                                f"until getent hosts {self._tfdata_dispatcher_service_name}; do echo 'Waiting for dispatcher service...'; sleep 2; done; {python_command}",
                            ]

                            container["command"] = wait_command
                            container["args"] = []
                            _LOGGER.info(
                                f"Wrapped worker command with DNS wait for: {self._tfdata_dispatcher_service_name}"
                            )
                    # Handle Service specific fields
                    elif kind == "Service":
                        new_name = (
                            self._tfdata_dispatcher_service_name if is_dispatcher else self._tfdata_worker_service_name
                        )
                        resource["metadata"]["name"] = new_name

                        # Add persistent service labels
                        if "labels" not in resource["metadata"]:
                            resource["metadata"]["labels"] = {}
                        resource["metadata"]["labels"].update(
                            {
                                "app.kubernetes.io/name": "tf-data-service",
                                "app.kubernetes.io/component": "dispatcher" if is_dispatcher else "worker",
                                "exec-ml/tf-data-service": "persistent",
                            }
                        )

                        # Update selector to match tf-data-service label from all deployments
                        resource["spec"]["selector"] = {"tf-data-service": "dispatcher" if is_dispatcher else "worker"}
                        _LOGGER.info(f"Updated service selector for {new_name}: {resource['spec']['selector']}")

        return {"resources": tfdata_resources}

    def _set_tfdata_container_image(self, tfdata_yaml: dict):
        """Set tf.data deployment resources container image."""
        container_image: Union[str, None] = self._get_container_image()
        if container_image:
            tfdata_yaml["spec"]["template"]["spec"]["containers"][0]["image"] = container_image
            _LOGGER.info(f"Updated tf.data service container image to: {container_image}")

    def _create_tfdata_resources(self, tfdata_yaml_body: dict):
        """Create tf.data service resources using the imported and updated YAML."""
        apps_client = Pod.get_k8s_apps_client(self._k8s_apps_client)
        core_client = Pod.get_k8s_core_client(self._k8s_core_client)

        for resource in tfdata_yaml_body["resources"]:
            if not resource:
                continue

            resource_name = resource["metadata"]["name"]
            resource_kind = resource["kind"]

            _LOGGER.info(f"Creating {resource_kind} {resource_name} in namespace {self._namespace}")

            try:
                if resource_kind == "Deployment":
                    apps_client.create_namespaced_deployment(namespace=self._namespace, body=resource)
                    _LOGGER.info(f"Deployment {resource_name} created successfully")

                elif resource_kind == "Service":
                    # Check if service already exists before creating
                    try:
                        existing_service = core_client.read_namespaced_service(
                            name=resource_name, namespace=self._namespace
                        )
                        _LOGGER.info(
                            f"Persistent Service {resource_name} already exists - reusing persistent service {existing_service}"
                        )
                        continue  # Skip creation if service exists
                    except client.rest.ApiException as e:
                        if e.status == 404:
                            # Service doesn't exist, create it
                            core_client.create_namespaced_service(namespace=self._namespace, body=resource)
                            _LOGGER.info(f"Service {resource_name} created successfully")
                        else:
             #               _LOGGER.error(f"Error checking service {resource_name}: {e}")
                            raise

            except client.rest.ApiException as e:
                if e.status == 409:  # Resource already exists
                    _LOGGER.info(f"{resource_kind} {resource_name} already exists")
                else:
             #       _LOGGER.error(f"Failed to create {resource_kind} {resource_name}: {e}")
                    raise RuntimeError(f"Exception when calling {resource_kind}Api: {e}")
            except Exception as e:
           #     _LOGGER.error(f"Unexpected error creating {resource_kind} {resource_name}: {e}")
                raise

    def _prepare_job_yaml(self, exec_properties: dict[str, Any], args: list[str]) -> dict:
        """Prepare Job yaml"""
        yaml_file_name: str = exec_properties["module"].lower().replace("_", "-") + "-" + self._yaml_name
        try:
            yaml_path: str = properties.get_file_path(file_name=yaml_file_name)
        except ValueError:
            _LOGGER.warning(f"Yaml file '{yaml_file_name}' not found, using default yaml: {self._yaml_name}")
            yaml_path: str = properties.get_file_path(file_name=self._yaml_name)

        with open(yaml_path) as f:
            job_yaml: dict = yaml.safe_load(f)

        # job name should be unique to each run
        job_yaml["metadata"]["name"] = self._job_name

        # clean up all pods after job completion
        if "runPolicy" not in job_yaml["spec"]:
            job_yaml["spec"]["runPolicy"] = {}
        job_yaml["spec"]["runPolicy"].update({"cleanPodPolicy": "All"})

        container_image: Union[str, None] = self._get_container_image()

        for replica in job_yaml["spec"][self._spec_name]:
            # add labels to pods (minor hack to get around version mismatch between python sdk and deployment)
            if "label" not in job_yaml["spec"][self._spec_name][replica]["template"]["metadata"]:
                job_yaml["spec"][self._spec_name][replica]["template"]["metadata"]["labels"] = {}
            job_yaml["spec"][self._spec_name][replica]["template"]["metadata"]["labels"].update(
                {
                    "group-name": "kubeflow.org",
                    "job-name": self._job_name,
                    "replica-type": replica.lower(),
                }
            )

            # restart behavior of pods
            if replica in self._restart_policy:
                job_yaml["spec"][self._spec_name][replica]["restartPolicy"] = self._restart_policy[replica]

            # extend entrypoint args
            job_yaml["spec"][self._spec_name][replica]["template"]["spec"]["containers"][0]["command"].extend(args)

            # override container image
            if container_image:
                job_yaml["spec"][self._spec_name][replica]["template"]["spec"]["containers"][0]["image"] = (
                    container_image
                )

        return job_yaml

    def _get_container_image(self) -> Union[str, None]:
        """Get container image of current pod"""
        if properties.get_boolean_property(
            section="KFPipeline",
            property_name=f"override.{self._yaml_name}.image",
            default_value="True",
        ):
            _LOGGER.info("Overriding replica container images")
            for container in Pod.get_current_spec().containers:
                if container.name == "main":
                    return container.image
            raise CurrentPodContainerImageNotFoundException(
                "Unable to determine container image from current pod specification."
            )

    def _get_job_name(self) -> str:
        tenant_id: str = self.get_tenant_id().replace("-", "")
        pipeline_name: str = self.get_pipeline_name().lower().replace(tenant_id, "").replace("trainingpipeline", "")
        job_name = f"{pipeline_name}-{self.get_tfx_node_id()}-{self.get_run_id()}"
        return job_name

    def _is_job_running(self) -> bool:
        jobs: dict = self._client.get(namespace=self._namespace, kind=self._kind)
        if any((item["metadata"]["name"] == self._job_name) for item in jobs["items"]):
            _LOGGER.info(f"Job {self._job_name} already exists.")
            if self._client.is_job_running(
                name=self._job_name,
                namespace=self._namespace,
                kind=self._kind,
            ):
                _LOGGER.info(f"Job {self._job_name} is in Running status.")
                return True
            else:
                _LOGGER.info(f"Job {self._job_name} is not in Running status, deleting it.")
                self._client.delete(
                    self._job_name,
                    namespace=self._namespace,
                    kind=self._kind,
                )
                return False
        return False

    def _launch_job(self):
        """Launch Job"""
        response: dict = self._client.create(self._job_body, namespace=self._namespace)
        _LOGGER.info(f"response: {response}")

    def _track_job(self):
        """Track Job"""
        while True:
            status: str = self._get_status()
            _LOGGER.info(f"Job | status: {self._job_name} | {status}")

            # wait for job to start
            if status in ["", "Created", "Restarting"]:
                _LOGGER.info(f'Job {self._job_name} is in "{status}" status, waiting for run.')
                self._client.wait_for_condition(
                    self._job_name,
                    expected_condition=["Running", "Succeeded", "Failed"],
                    namespace=self._namespace,
                    kind=self._kind,
                )
                continue

            # break away from loop if job succeeded
            elif status == "Succeeded":
                _LOGGER.info(f"Job {self._job_name} completed successfully")
                return

            # restart job if it failed (and we haven't exceeded fail tolerance)
            elif status == "Failed":
                if self._check_fail_mode():
                    self._restart_job()
                    continue
                else:
                    raise ValueError("TFJob failed due to chief failure. Check chief logs above")

            # sleep for 5 minutes if job is running
            elif status == "Running":
                time.sleep(60 * 5)
                _LOGGER.info(f"Training job '{self._job_name}' still running")

            # raise error if job is in unknown status
            else:
                raise ValueError(f"Unknown Job status: {status}")

    def _get_status(self) -> str:
        """Get Job status"""
        status: str = self._client.get_job_status(
            name=self._job_name,
            namespace=self._namespace,
            kind=self._kind,
        )
        return status

    def _check_fail_mode(self) -> bool:
        """Check replica status and restart if it is not a chief/master failure"""
        response: dict = self._client.get(
            name=self._job_name,
            namespace=self._namespace,
            kind=self._kind,
        )
        _LOGGER.info(f"Replica statuses of Job: \n\n{response.get('status', {}).get('replicaStatuses', {})}")
        if response.get("status", {}).get("replicaStatuses", {}).get(self._main_replica, {}).get("failed", 0) == 0:
            _LOGGER.info(f"Job {self._job_name} status - Failed. {self._main_replica} pod active..")
            return True
        else:
            _LOGGER.info(f"Job {self._job_name} status - Failed. {self._main_replica} pod failed.")
            return False

    def _restart_job(self):
        """Restart Job"""
        if self._n_fails > self._fail_tolerance:
            raise ValueError(f"Maximum number of job failure exceeded: {self._n_fails}")
        _LOGGER.info(f"Restarting job {self._job_name}, attempt {self._n_fails + 1}")
        self._client.delete(
            name=self._job_name,
            namespace=self._namespace,
            kind=self._kind,
        )
        self._client.create(tfjob=self._job_body, namespace=self._namespace)
        self._n_fails += 1
