import logging
import os
from typing import Self, Tuple

import hydra
from hydra.utils import instantiate
from langchain.tools import tool
from omegaconf import DictConfig
from semistaticsim.groundtruth.simulator import Simulator
from langchain_core.callbacks import StdOutCallbackHandler

from concept_graphs.mapping.PerpetuaObjectMap import PerpetuaObjectMap
from semistaticsim_rollout import Agent

# Disable GPU memory pre-allocation to avoid OOM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from typing import Any, Dict

from typing import Union

from langchain.agents import create_agent

from concept_graphs.utils import set_seed


from concept_graphs.viz.server.ObjectMapServer import ObjectMapServer
from concept_graphs.viz.server.PerpetuaMapServer import PerpetuaMapServer

# A logger for this file
log = logging.getLogger(__name__)

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentPrinterCallback(BaseCallbackHandler):
    """
    Custom callback to print LLM reasoning and tool usage to the console.
    """

    def on_chat_model_start(self, serialized, messages, **kwargs):
        # Optional: Print "Thinking..." if you want to know it started
        pass

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Print the LLM's reasoning or text response."""
        # Access the actual text generated
        if response.generations and response.generations[0]:
            generation = response.generations[0][0]
            # Check if it's a ChatGeneration (standard for OpenAI)
            if hasattr(generation, "message"):
                content = generation.message.content
                # Only print if there is actual text (sometimes it's just a tool call)
                if content:
                    print(f"\n\033[1m[LLM Reasoning]\033[0m:\n{content}\n")
            # Fallback for standard generation
            elif generation.text:
                print(f"\n\033[1m[LLM Reasoning]\033[0m:\n{generation.text}\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Print which tool is being called and with what inputs."""
        tool_name = serialized.get("name")
        print(f"\033[94m[Tool Call]\033[0m: {tool_name}")
        print(f"\033[94m[Arguments]\033[0m: {input_str}")

    def on_tool_end(self, output, **kwargs):
        """Print the result from the tool."""
        print(f"\033[92m[Tool Output]\033[0m: {str(output)}\n")


class PerpetuaAgent(Agent):
    def __init__(self, sim: Simulator, perpetua_map_server: PerpetuaMapServer):
        super().__init__(sim)
        self.perpetua_map_server = perpetua_map_server
        self.llm_model = "openai:gpt-5"
        self.llm_agent = create_agent(model=self.llm_model, tools=self.tools())

    @property
    def perpetua_model(self) -> PerpetuaObjectMap:
        return self.perpetua_map_server.toolbox.object_map

    def tools(self):
        @tool
        def predict_object_receptacle(pickupable_id) -> Tuple[str, Dict[str, float]]:
            """

            Predicts the current receptacle for an object.
            The results of this tool should be trusted above any semantic prior you might have!
            If the most likely receptacle reported by this tool seems nonsensical, trust it anyway. It has learned from real data.

            Args:
                pickupable_id: A pickupable object's id. Must match exactly.

            Returns:
                argmax_receptacle (str): Most likely receptacle
                recptacle_weights (Dict[str, float]) Dict of probability weights for each receptacle]

            """
            return self.predict_object_receptacle(pickupable_id)

        @tool
        def go_to_receptacle(receptacle_id: str, pickupable_id: str):
            """

            Args:
                receptacle_id: The name of the target receptacle.
                looking_for_pickupable: The name of the pickupable we are looking for.

            Returns: True if the current state contains the pickupable, False otherwise.

            """
            return self.go_to_receptacle(
                receptacle_id, pickupable_id
            )  # .go_to_receptacle(receptacle_id, pickupable_id)

        return [predict_object_receptacle, go_to_receptacle]

    def query(self, query: str):
        system_message = {
            "role": "system",
            "content": f"""
You are an embodied LLM planner. 

Here are the IDs of the objects of interest: <{', '.join(self.sim.sss_data.pickupable_names)}>
Each object of interest will be in different <receptacles> through time.
The receptacles IDs are: <{', '.join(self.sim.sss_data.receptacle_names)}>

You cannot assume the current receptacle of any object of interest. 
You must use your prediction tool to figure out where the object currently is.

You must reason about the user query; some only require prediction, some only require navigation, some require both.
Do not go to an object unless explicitly necessary to fulfill the query.
""",
        }

        user_message = {"role": "user", "content": query}

        return self.llm_agent.invoke(
            {"messages": [system_message, user_message]},
            config={"callbacks": [AgentPrinterCallback()]},
        )

    @property
    def current_time(self):
        return float(self.sim.sss_data.self_at_current_time._timestamp)

    def _predict_object_receptacle(
        self, pickupable: str, current_time: float = None
    ) -> Dict[str, float]:
        """

        Args:
            pickupable_id: A pickupable object's id. Must match exactly.

        Returns: Tuple[Most likely receptacle, Dict of probability weights for each receptacle]

        """
        pickupable_id = self.resolve_query_into_pickupable(pickupable)
        prediction = self.perpetua_model.object_predict(pickupable_id, current_time)
        sorted_keys = sorted(prediction, key=prediction.get, reverse=True)
        return sorted_keys[0], prediction

    def update(self, observation: Dict[str, Any]):
        return None  # todo self.perpetua_model.object_update(current_time, pickupable)

    def found_pickupable(
        self, receptacle_name: str, pickupable_name: str, observation: Dict[str, Any]
    ) -> Union[bool, None]:
        """

        Args:
            observation: An observation of the current state.

        Returns: True if the current state contains the pickupable. False if the receptacle doesn't have the pickupable. None if no information was gained.

        """
        from semistaticsim.datawrangling.sssd import GeneratedSemiStaticData

        apn: GeneratedSemiStaticData = observation["point_apn"]

        vector_for_pickupable = apn._assignment[
            apn.pickupable_names.index(pickupable_name)
        ]
        if (vector_for_pickupable == 1).any():
            return True

        try:
            if vector_for_pickupable[apn.receptacle_names.index(receptacle_name)] >= 0:
                return False
        except:
            pass

        return None


@hydra.main(version_base=None, config_path="conf", config_name="map_server")
def main(cfg: DictConfig):
    from semistaticsim.keyboardcontrol import main_skillsim
    from pathlib import Path

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # todo when the config change is implemented for priv.
    # change overrides to just `config.mode.runfunc="rollout"`
    # 1. Load config from file
    # 2. Set cfg.mode.runfunc="rollout"
    # 3. set cfg.mode.get_simulator_instance=True
    # 4. also can tweak the start_at
    cfg_path = Path(cfg.data_dir) / cfg.scene
    gt_cfg_path = Path(cfg.data_dir).parent / "groundtruth"

    sim = main_skillsim.main(
        config_path=str(cfg_path),
        overrides=[
            "mode.runfunc=rollout",
            "mode.get_simulator_instance=True",
            f"scene={str(gt_cfg_path)}",
        ],
    )

    set_seed(cfg.seed)
    server: Union[ObjectMapServer | PerpetuaMapServer] = instantiate(cfg.server)
    log.info(f"Loading map with a total of {len(server.object_map)} objects")
    perpetua_agent = PerpetuaAgent(sim, server)

    """
    Note: the loop below is meant for open-vocab semistatic llm planning.
    For closed-vocab, use the existing functions like so:
    1. get target pickupable ID
    2. target_receptacle, _ = agent.predict_object_receptacle(pickupable_id)
    3. agent.go_to_receptacle(target_receptacle, pickupable_id)
    """

    print("Terminal chatbot started.")
    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            response = perpetua_agent.query(user_input)

            # Handle cases where the agent returns non-string output
            if isinstance(response, str):
                print(response)
            else:
                print(str(response))

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            raise e


if __name__ == "__main__":
    import os

    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import jax

    jax.config.update("jax_platform_name", "cpu")

    main()
