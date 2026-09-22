# **************************************************************************
# * main.py -- instance twin entrypoint
# **************************************************************************
import logging
import os
import time
from importlib.resources import files

# --- ADD THIS MONKEY PATCH -------------
import paho.mqtt.client as mqtt

from instance_twin.instance_comms import InstanceNetworkNode
from instance_twin.gui.twin_gui import start_gui

# Save the original subscribe method
_original_subscribe = mqtt.Client.subscribe

# Create a safe wrapper that catches and discards unexpected kwargs (like 'isService')
def _safe_subscribe(self, topic, qos=0, options=None, properties=None, **kwargs):
    return _original_subscribe(self, topic, qos=qos, options=options, properties=properties)

# Override the library method globally
mqtt.Client.subscribe = _safe_subscribe

# ------------------------------------------

from core_msgs.topic_contract import load_topic_config
from core_msgs.utils.utils import load_config, format_nested_strings

from instance_twin.instance_base import InstanceTwin

TICK_HZ = float(os.environ.get("TWIN_TICK_HZ", "10"))
MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "localhost")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
NAMESPACE = os.environ.get("TWIN_NAMESPACE", "default_ns")
TWIN_NAME = os.environ.get("TWIN_NAME", os.environ.get("HOSTNAME", "InstanceTwin"))
DRIVE_AGENT = os.environ.get("TWIN_DRIVE_AGENT", "0").strip().lower() in ("1", "true", "yes")
GUI_ENABLED = os.environ.get("TWIN_GUI", "1").strip().lower() in ("1", "true", "yes")
GUI_PORT = int(os.environ.get("TWIN_GUI_PORT", "8081"))

logger = logging.getLogger(__name__)


def main() -> None:

    # Comms configs
    flex_config_path = files("instance_twin").joinpath("config", "config.yaml")
    flex_config_dict = load_config(str(flex_config_path))

    formatted_flex_config = format_nested_strings(
        flex_config_dict,
        mqtt_address=MQTT_BROKER_HOST,
        redis_address=REDIS_HOST,
        node_id=TWIN_NAME,
        namespace=NAMESPACE,
    )
    global_topic_path = files("instance_twin").joinpath("config", "global_topic_config.yaml")
    global_topic_dict = load_topic_config(str(global_topic_path))

    aggregate_topic_path = files("instance_twin").joinpath("config", "aggregate_topic_config.yaml")
    aggregate_topic_dict = load_topic_config(str(aggregate_topic_path))

    empty_comm_matrix_path = files("instance_twin").joinpath("config", "emptyCommMatrix.yaml")



    twin = InstanceTwin(
        name=TWIN_NAME,
        namespace=NAMESPACE,
        loop_freq=TICK_HZ,
        drive_agent=DRIVE_AGENT,
    )

    network_node = InstanceNetworkNode(
        flex_config=formatted_flex_config,
        global_topic_dict=global_topic_dict,
        aggregate_topic_dict=aggregate_topic_dict,
        comm_matrix_path=str(empty_comm_matrix_path),
        twin=twin,
        loop_freq=TICK_HZ
    )

    network_node.spin()

    twin.setup_transport(network_node)


    gui = start_gui(twin, port=GUI_PORT) if GUI_ENABLED else None
    if gui:
        print(f"Twin viewer: http://172.0.0.1:{GUI_PORT}/")

    print(f"Instance twin '{TWIN_NAME}' waiting for instantiate on namespace={NAMESPACE}")


    try:
        logger.info(f"Started Instance Twin: {TWIN_NAME}")
        while True:
            time.sleep(1)  # Sleep to avoid busy-waiting
    except KeyboardInterrupt:
        twin.shutdown()
        logger.warning("\nShutdown signal received. Exiting...")


if __name__ == "__main__":
    main()