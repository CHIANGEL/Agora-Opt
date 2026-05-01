from .answer_and_dataset import *
import shutil
from .parameters import get_params
from .constraint import get_constraints
from .constraint_model import get_constraint_formulations
from .target_code import get_codes
from .generate_code import generate_code
from .utils import load_state, save_state, Logger, get_response
from .objective import get_objective
from .objective_model import get_objective_formulation
from .execute_code import execute_and_debug
import json
import traceback
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
import argparse
from .debate_prompt import *


ERROR_CORRECTION = True
# MODEL = "gemini-2.5-pro"
DEFAULT_LABELS = {"types": ["Mathematical Optimization"], "domains": ["Operations Management"]}
MAX_DEBATE_ROUNDS = 3
# TEMPERATURE = 0.01  # Reproducibility-oriented default.
# Per-team temperature settings.
# TEMPERATURE_A = 0.2
# TEMPERATURE_B = 0.5


# revision_history_path log format:
# "Round {i}:\n"
# "Comment from A to B: {json.dumps(commentA)}\n"
# "Reply from B: {replyB}\n"
# f"Comment from B to A: {json.dumps(commentB)}\n"
# f"Reply from A: {replyA}\n\n"
def get_revision_history_json_from_log(revision_history_path):
    """Parse the plain-text revision log into JSON for prompt construction.

    Args:
        revision_history_path (str): Path to the revision history log file.
    """
    with open(revision_history_path, "r") as f:
        lines = f.readlines()
    revision_history = {}
    current_round = None
    for line in lines:
        line = line.strip()
        if line.startswith("Round"):
            current_round = line.split(" ")[1].strip(":")
            revision_history[current_round] = {}
        elif line.startswith("Comment from A to B:"):
            if current_round is not None:
                commentA_str = line.split("Comment from A to B:")[-1].strip()
                try:
                    commentA = json.loads(commentA_str)
                except:
                    commentA = commentA_str
                revision_history[current_round]["commentA"] = commentA
        elif line.startswith("Reply from B:"):
            if current_round is not None:
                replyB = line.split("Reply from B:")[-1].strip()
                revision_history[current_round]["replyB"] = replyB
        elif line.startswith("Comment from B to A:"):
            if current_round is not None:
                commentB_str = line.split("Comment from B to A:")[-1].strip()
                try:
                    commentB = json.loads(commentB_str)
                except:
                    commentB = commentB_str
                revision_history[current_round]["commentB"] = commentB
        elif line.startswith("Reply from A:"):
            if current_round is not None:
                replyA = line.split("Reply from A:")[-1].strip()
                revision_history[current_round]["replyA"] = replyA
                
    return revision_history


def cut_revision_history(revision_history_json, actor, action, if_real = True):
    if not if_real:
        return revision_history_json
    
    assert actor in ["A", "B"], "actor must be A or B"
    assert action in ["comment", "reform"], "action must be comment or reform"
    
    """
    For comment generation, expose the actor's previous comments and the other
    side's replies. For reformulation, expose the other side's comments and the
    actor's previous replies.
    """
    revision_history_cut = {}
    for round in revision_history_json:
        revision_history_cut[round] = {}
        if action == "comment":
            if actor == "A":
                if "commentA" in revision_history_json[round]:
                    revision_history_cut[round]["commentA"] = revision_history_json[round]["commentA"]
                if "replyB" in revision_history_json[round]:
                    revision_history_cut[round]["replyB"] = revision_history_json[round]["replyB"]
            elif actor == "B":
                if "commentB" in revision_history_json[round]:
                    revision_history_cut[round]["commentB"] = revision_history_json[round]["commentB"]
                if "replyA" in revision_history_json[round]:
                    revision_history_cut[round]["replyA"] = revision_history_json[round]["replyA"]  
                    
        elif action == "reform":
            if actor == "A":
                if "commentB" in revision_history_json[round]:
                    revision_history_cut[round]["commentB"] = revision_history_json[round]["commentB"]
                if "replyA" in revision_history_json[round]:
                    revision_history_cut[round]["replyA"] = revision_history_json[round]["replyA"]  
            elif actor == "B":
                if "commentA" in revision_history_json[round]:
                    revision_history_cut[round]["commentA"] = revision_history_json[round]["commentA"]
                if "replyB" in revision_history_json[round]:
                    revision_history_cut[round]["replyB"] = revision_history_json[round]["replyB"] 
    return revision_history_cut
    
    
    

def choose_prompt(version):
    if int(version) == 1:
        return REFORMULATION_PROMPT_BASE1, CRITIC_PROMPT_BASE1
    elif int(version) == 2:
        return REFORMULATION_PROMPT_BASE2, CRITIC_PROMPT_BASE2
    else:
        raise ValueError("Invalid prompt version")


def get_comment(revision_history_flag, desc, state_alice, state_bob, model, temperature, revision_history_json, version):
    if revision_history_flag:
        prompt = choose_prompt(version)[1]
        comment = extract_comment(get_response(
            prompt.format(
                description=desc,
                state_alice=state_alice,
                state_bob=state_bob,
                revision_history=json.dumps(revision_history_json, indent=4)
            ),
            model=model, temperature=temperature
        ))
    else:
        prompt = CRITIC_PROMPT_BASE0
        comment = extract_comment(get_response(
            prompt.format(
                description=desc,
                state_alice=state_alice,
                state_bob=state_bob,
            ),
            model=model, temperature=temperature
        ))
        
    return comment


def get_reformulation_and_reply(revision_history_flag, desc,comments_bob, state_alice, model, temperature, revision_history_json, version):
    if revision_history_flag:
        prompt = choose_prompt(version)[0]
        response = get_response(
            prompt.format(
                description=desc,
                comments_bob=comments_bob,
                state_alice=state_alice,
                revision_history=json.dumps(revision_history_json), indent=4),
            model=model, temperature=temperature
        )
    else:
        prompt = REFORMULATION_PROMPT_BASE0
        response = get_response(
            prompt.format(
                description=desc,
                comments_bob=comments_bob,
                state_alice=state_alice,
            ),
            model=model, temperature=temperature
        )
    
    prompt = extract_reformulation(response)
    if revision_history_flag:
        reply = extract_reply(response)
    else:
        reply = ""
    return prompt, reply


def house_remove(problem_path, modelA_problem_path, modelB_problem_path,):
    try:
        with open(os.path.join(modelA_problem_path, "output_solution.txt"), "r") as f:
            answerA = f.read().strip()
    except Exception as e:
        print(f"Error reading output_solution.txt from {modelA_problem_path}: {e}")
        answerA = None
    try:
        with open(os.path.join(modelB_problem_path, "output_solution.txt"), "r") as f:
            answerB = f.read().strip()
    except Exception as e:
        print(f"Error reading output_solution.txt from {modelB_problem_path}: {e}")
        answerB = None

    with open(os.path.join(modelA_problem_path, "data.json"), "r") as f:
        dataA = json.load(f)
    with open(os.path.join(modelB_problem_path, "data.json"), "r") as f:
        dataB = json.load(f)
        
    with open(os.path.join(modelA_problem_path, "params.json"), "r") as f:
        paramsA = json.load(f)
    with open(os.path.join(modelB_problem_path, "params.json"), "r") as f:
        paramsB = json.load(f)

    with open(os.path.join(modelA_problem_path, "state_6_code.json"), "r") as f:
        stateA = json.load(f)
    with open(os.path.join(modelB_problem_path, "state_6_code.json"), "r") as f:
        stateB = json.load(f)

    # Copy these artifacts into the team_A and team_B subdirectories.
    with open(os.path.join(problem_path, "team_A", "state_6_code.json"), "w") as f:
        json.dump(stateA, f, indent=4)  
    with open(os.path.join(problem_path, "team_B", "state_6_code.json"), "w") as f:
        json.dump(stateB, f, indent=4)
        
    with open(os.path.join(problem_path, "team_A", "data.json"), "w") as f:
        json.dump(dataA, f, indent=4)
    with open(os.path.join(problem_path, "team_B", "data.json"), "w") as f:
        json.dump(dataB, f, indent=4)
    
    with open(os.path.join(problem_path, "team_A", "params.json"), "w") as f:
        json.dump(paramsA, f, indent=4)
    with open(os.path.join(problem_path, "team_B", "params.json"), "w") as f:
        json.dump(paramsB, f, indent=4)
        
        
    try:
        answerA = float(answerA)
    except:
        answerA = None
    try:
        answerB = float(answerB)
    except:
        answerB = None
    return answerA, answerB


def get_answer_from_description_single(desc, problem_path, model, temperature, start_stage = 0, coding_model = None) -> float:  
    # assert start_stage in range(0, 8), "start_stage must be between 0 and 7"
    # stage0: Init
    logger_results = Logger(os.path.join(problem_path, "log_results.txt"))
    logger = Logger(os.path.join(problem_path, "log.txt"))
    
    if start_stage == 999:
        if os.path.exists(os.path.join(problem_path, "output_solution.txt")):
            print(f"Problem {problem_path} already solved, skipping.")
            with open(os.path.join(problem_path, "output_solution.txt"), "r") as f:
                output = f.read().strip()
            return output
        elif os.path.exists(os.path.join(problem_path, "code_3.py")):
            print(f"Problem {problem_path} already has code_3.py, skipping to execution.")
    
        
        # Infer the next stage from the existing state files.
        print("Inferring start_stage from existing state files...")
        state_files = [f for f in os.listdir(problem_path) if f.startswith("state_") and f.endswith(".json")]
        if not state_files:
            start_stage = 0
        else:
            stages = [int(f.split("_")[1]) for f in state_files]
            start_stage = max(stages) + 1
    
    if start_stage == 0:
        logger.reset()
        logger_results.reset()
    
    if coding_model is None:
        coding_model = model
    
    # stage1: create state and params
    if start_stage <= 1:
        params = get_params(desc, model=model, temperature=temperature)
        with open(os.path.join(problem_path, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
        with open(os.path.join(problem_path, "params.json"), "r") as f:
            params = json.load(f)
        data = {}
        for key in params:
            data[key] = params[key]["value"]
            del params[key]["value"]
        # save the data file in the run_dir
        with open(os.path.join(problem_path,"data.json"), "w") as f:
            json.dump(data, f, indent=4)

        state = {"description": desc, "parameters": params}
        save_state(state, os.path.join(problem_path, "state_1_params.json"))
    
    # stage2: get objective
    if start_stage <= 2:
        state = load_state(os.path.join(problem_path, "state_1_params.json"))
        objective = get_objective(
            state["description"],
            state["parameters"],
            check=ERROR_CORRECTION,
            logger=logger,
            model=model,
            labels=DEFAULT_LABELS,
            temperature=temperature,
        )
        state["objective"] = objective
        save_state(state, os.path.join(problem_path, "state_2_objective.json"))
    
    # stage3: get constraints
    if start_stage <= 3:
        state = load_state(os.path.join(problem_path, "state_2_objective.json"))
        constraints = get_constraints(
            state["description"],
            state["parameters"],
            check=ERROR_CORRECTION,
            logger=logger,
            model=model,
            labels=DEFAULT_LABELS,
            temperature=temperature,
        )
        state["constraints"] = constraints
        save_state(state, os.path.join(problem_path, "state_3_constraints.json"))
    
    # stage4: get constraint formulations
    if start_stage <= 4:
        state = load_state(os.path.join(problem_path, "state_3_constraints.json"))
        constraints, variables = get_constraint_formulations(
            state["description"],
            state["parameters"],
            state["constraints"],
            check=ERROR_CORRECTION,
            logger=logger,
            model=model,
            labels=DEFAULT_LABELS,
            temperature=temperature,
        )
        state["constraints"] = constraints
        state["variables"] = variables
        save_state(state, os.path.join(problem_path, "state_4_constraints_modeled.json"))
        
    # stage5: get objective formulation
    if start_stage <= 5:
        state = load_state(os.path.join(problem_path, "state_4_constraints_modeled.json"))
        objective = get_objective_formulation(
            state["description"],
            state["parameters"],
            state["variables"],
            state["objective"],
            model=model,
            check=ERROR_CORRECTION,
            labels=DEFAULT_LABELS,
            temperature=temperature,
        )
        state["objective"] = objective
        save_state(state, os.path.join(problem_path, "state_5_objective_modeled.json"))
        
    # stage6: get code
    if start_stage <= 6:
        state = load_state(os.path.join(problem_path, "state_5_objective_modeled.json"))
        constraints, objective = get_codes(
            state["description"],
            state["parameters"],
            state["variables"],
            state["constraints"],
            state["objective"],
            model=coding_model,
            check=ERROR_CORRECTION,
            temperature=temperature,
        )
        state["constraints"] = constraints
        state["objective"] = objective
        save_state(state, os.path.join(problem_path, "state_6_code.json"))
        
    # stage7: run the code and debug
    if start_stage <= 7:
        state = load_state(os.path.join(problem_path, "state_6_code.json"))
        generate_code(state, problem_path)
        output = execute_and_debug(state, 
                                model=coding_model, 
                                temperature=temperature,
                                dir=problem_path, 
                                logger=logger
                                )
        
        logger_results.log(f"Question id: {problem_path}, Final result: {output}")
        return output
    


def get_answer_from_description_debate(desc: str, problem_path, modelA = "gpt-4o", temperatureA = 0.01, modelB = "gemini-2.5-pro", temperatureB = 0.01, start_stage=0, modelA_problem_path = None, modelB_problem_path = None, confident_mode = "least_change", revision_history = False, coding_model = None, prompt_version = 2) -> float:
    """
    Debate-mode wrapper around the single-agent pipeline.

    Workflow:
    1. Run two single-agent solutions and compare their answers.
    2. If they agree, return the answer directly.
    3. Otherwise enter debate mode:
       - each side comments on the other side's state
       - each side reformulates its own state
       - the code-generation / execution stage is rerun
       - answers are compared again and the process repeats

    start_stage:
    0: run from scratch
    1: reuse existing single-agent results as debate inputs
    2: read the final outputs from the two team folders directly
    """
    debate_logger = Logger(os.path.join(problem_path, "log_debate.txt"))
    converge_logger = Logger(os.path.join(problem_path, "log_converge.txt"))

    
    # Input validation.
    assert start_stage in range(0, 3), "start_stage must be between 0 and 2"
    assert confident_mode in ["least_change", "fixed_A", "fixed_B"], "confident_mode must be one of ['least_change', 'fixed_A', 'fixed_B']"
    
    if revision_history:
        revision_logger = Logger(os.path.join(problem_path, "log_revision_history.txt"))
        revision_logger.reset()
    

    if start_stage < 2:
        debate_logger.reset()
        converge_logger.reset()
        if start_stage == 0:  # Generate the initial solutions.
            try:
                answerA = get_answer_from_output(get_answer_from_description_single(desc, os.path.join(problem_path, "team_A"), model=modelA, temperature=temperatureA, start_stage=0))
            except Exception as e:
                debate_logger.log(f"Error in team A initial solution: {e}")
                print(f"Error in team A initial solution: {e}")
                answerA = None
                
            try:
                answerB = get_answer_from_output(get_answer_from_description_single(desc, os.path.join(problem_path, "team_B"), model=modelB, temperature=temperatureB, start_stage=0))
            except Exception as e:
                debate_logger.log(f"Error in team B initial solution: {e}")
                print(f"Error in team B initial solution: {e}")
                answerB = None
                
        elif start_stage == 1:
            # Copy the key single-agent artifacts into the two debate team folders.
            answerA, answerB = house_remove(problem_path, modelA_problem_path, modelB_problem_path)
        
        # Compare initial solutions and decide whether debate is needed.
        if answerA is None and answerB is None:
            debate_logger.log(f"Both teams failed to get initial solutions for problem: {problem_path}")
            print(f"Both teams failed to get initial solutions for problem: {problem_path}")
            return None

        # elif answerA is False:
        #     answer = answerB
        #     debate_logger.log(f"Team A failed to get initial solution, using Team B's solution: {answerB}")
        #     print(f"Team A failed to get initial solution, using Team B's solution: {answerB}")
        #     return answer
        # elif answerB is False:
        #     answer = answerA
        #     debate_logger.log(f"Team B failed to get initial solution, using Team A's solution: {answerA}")
        #     print(f"Team B failed to get initial solution, using Team A's solution: {answerA}")
        #     return answer
            
        elif converge(answerA, answerB):
            debate_logger.log(f"Initial solutions converged: {answerA}")
            print("Initial answers converge")
            answer = (answerA + answerB) / 2
            return answer
        
        else:  # Enter debate mode.
            print("START DEBATING!!!")
            debate_logger.log(f"Entering debate mode with initial answers: A={answerA}, B={answerB}")
            converge_logger.log(f"A0: {answerA}, B0: {answerB}")
            if revision_history:
                revision_history_path = os.path.join(problem_path, "log_revision_history.txt")
            else:
                revision_history_path = None
            for i in range(1, MAX_DEBATE_ROUNDS+1):
                if converge(answerA, answerB):
                    debate_logger.log(f"Debate converged after {MAX_DEBATE_ROUNDS - i} rounds: {answerA}")
                    print(f"Debate converged after {MAX_DEBATE_ROUNDS - i} rounds: {answerA}")
                    answer = (answerA + answerB) / 2
                    return answer 
                # If the answers still disagree, generate comments for this round.
                if not revision_history:
                    revision_history_json = None
                else:
                    revision_history_json = get_revision_history_json_from_log(revision_history_path)
                
                try:
                    commentA = get_comment(
                        revision_history_flag = revision_history,
                        desc=desc,
                        state_alice=json.dumps(load_state(os.path.join(problem_path, "team_A", "state_6_code.json")), indent=4),
                        state_bob=json.dumps(load_state(os.path.join(problem_path, "team_B", "state_6_code.json")), indent=4),
                        model=modelA, temperature=temperatureA, revision_history_json=cut_revision_history(revision_history_json, actor="A", action="comment") if revision_history else None,
                        version=prompt_version,
                    )
                    commentB = get_comment(
                        revision_history_flag = revision_history,
                        desc=desc,
                        state_alice=json.dumps(load_state(os.path.join(problem_path, "team_B", "state_6_code.json")), indent=4),
                        state_bob=json.dumps(load_state(os.path.join(problem_path, "team_A", "state_6_code.json")), indent=4),
                        model=modelB, temperature=temperatureB, revision_history_json=cut_revision_history(revision_history_json, actor="B", action="comment") if revision_history else None,
                        version=prompt_version,
                    )
                    print("COMMENT A: ", commentA)
                    print("COMMENT B: ", commentB)
                except Exception as e:
                    debate_logger.log(f"Error generating comments: {e}")
                    print(f"Error generating comments: {e}")
                    continue
                debate_logger.log(f"Round {i+1} comments: A={commentA}, B={commentB}")
                
                try:
                    reformulated_stateA, replyA = get_reformulation_and_reply(
                        revision_history_flag = revision_history,
                        desc=desc,
                        comments_bob=commentB,
                        state_alice=json.dumps(load_state(os.path.join(problem_path, "team_A", "state_6_code.json")), indent=4),
                        model=modelA, temperature=temperatureA, revision_history_json =cut_revision_history(revision_history_json, actor="A", action="reform") if revision_history else None,
                        version=prompt_version,
                    )
                    reformulated_stateB, replyB = get_reformulation_and_reply(
                        revision_history_flag = revision_history,
                        desc=desc,
                        comments_bob=commentA,
                        state_alice=json.dumps(load_state(os.path.join(problem_path, "team_B", "state_6_code.json")), indent=4),
                        model=modelB, temperature=temperatureB,
                        revision_history_json = cut_revision_history(revision_history_json, actor="B", action="reform") if revision_history else None,
                        version=prompt_version,
                    )
                    # Record revision history in a text log for later prompt reuse.
                    if revision_history:
                        with open(os.path.join(problem_path, "log_revision_history.txt"), "a") as f:
                            f.write(f"Round {i}:\n")
                            f.write(f"Comment from A to B: {json.dumps(commentA)}\n")
                            f.write(f"Reply from B: {replyB}\n")
                            f.write(f"Comment from B to A: {json.dumps(commentB)}\n")
                            f.write(f"Reply from A: {replyA}\n\n")
                            
                    
                    print("REFORMULATION AND RELPY COMPLETED")
                except Exception as e:
                    debate_logger.log(f"Error reformulating states: {e}")
                    print(f"Error reformulating states: {e}. Skipping this round.")
                    continue
                
                # Save the new states and archive the previous state_6 snapshots.
                try:
                    shutil.copy(os.path.join(problem_path, "team_A", "state_6_code.json"), os.path.join(problem_path, "team_A", f"round_{i+1}_A.json"))
                    shutil.copy(os.path.join(problem_path, "team_B", "state_6_code.json"), os.path.join(problem_path, "team_B", f"round_{i+1}_B.json"))
                    
                    with open(os.path.join(problem_path, "team_A", "state_5_objective_modeled.json"), "w") as f:
                        json.dump(reformulated_stateA, f, indent=4)
                    with open(os.path.join(problem_path, "team_B", "state_5_objective_modeled.json"), "w") as f:
                        json.dump(reformulated_stateB, f, indent=4)
                    debate_logger.log(f"Round {i+1} reformulated states saved. Updating state files for next round.")
                except Exception as e:
                    debate_logger.log(f"Error saving reformulated states: {e}")
                    print(f"Error saving reformulated states: {e}")
                    continue
                
                # Rerun the downstream code-generation and execution stages.
                try:
                    if coding_model is None:
                        modelA_temp = modelA
                        modelB_temp = modelB
                    else:
                        modelA_temp = coding_model
                        modelB_temp = coding_model
                    outputA = get_answer_from_description_single(desc, os.path.join(problem_path, "team_A"), model=modelA_temp, start_stage=6, temperature=temperatureA)
                    outputB = get_answer_from_description_single(desc, os.path.join(problem_path, "team_B"), model=modelB_temp, start_stage=6, temperature=temperatureB)
                    
                    answerA = get_answer_from_output(outputA)
                    answerB = get_answer_from_output(outputB)
                    
                    converge_logger.log(f"A{i}: {answerA}, B{i}: {answerB}") 
                    
                except Exception as e:
                    debate_logger.log(f"Error in round {i+1} execution after reformulation: {e}")
                    print(f"Error in round {i+1} execution after reformulation: {e}")
                    continue
                
            converge_logger.log(f"A{i+1}: {answerA}, B{i+1}: {answerB}")
            print(f"Debate did not converge after {MAX_DEBATE_ROUNDS} rounds.") 
        
    elif start_stage == 2:  # Do not solve again; read the final team answers directly.
        answerA, answerB = get_debate_two_answer(problem_path)
    
    print(f"Final answers before confident choice: A={answerA}, B={answerB}")
    final_answer = get_confident_answer(
        answerA,
        answerB,
        confident_mode,
        debate_logger_path=os.path.join(problem_path, "log_debate.txt"),
        converge_logger_path=os.path.join(problem_path, "log_converge.txt"),
    )
    debate_logger.log(f"Final chosen answer: {final_answer} from A={answerA}, B={answerB} using mode {confident_mode}")
    print(f"Final chosen answer: {final_answer} from A={answerA}, B={answerB} using mode {confident_mode}")
    return final_answer


def get_debate_two_answer(problem_path):
    with open(os.path.join(problem_path, "log_debate.txt"), "r") as f:
        lines = f.readlines()
    line0 = lines[0].strip()
    if line0.startswith("Initial solutions converged:"):
        answer = float(line0.split(":")[-1].strip())
        print(f"Initial solutions converged: {answer}")
        return answer, answer
    
    # Read the last converge log line and recover the final A/B answers.
    with open(os.path.join(problem_path, "log_converge.txt"), "r") as f:
        lines = f.readlines()
    last_line = lines[-1].strip()
    parts = last_line.split(",")
    answerA = parts[0].split(":")[-1].strip()
    answerB = parts[1].split(":")[-1].strip()
    return answerA, answerB
    
    

def get_confident_answer(answerA, answerB, confident_mode, converge_logger_path, debate_logger_path):
    if confident_mode == "fixed_A":
        return answerA
    elif confident_mode == "fixed_B":
        return answerB
    elif confident_mode == "least_change":  # Choose the answer with the most stable history.
        if answerA is None and answerB is not None:
            return answerB
        elif answerB is None and answerA is not None:
            return answerA
        elif answerA is None and answerB is None:
            return None
        
        converge_logger = Logger(converge_logger_path)
        debate_logger = Logger(debate_logger_path)
        answersA = [line.split(",")[0].split(":")[-1].strip() for line in converge_logger.get_lines() if line.startswith(f"A")]
        answersB = [line.split(",")[1].split(":")[-1].strip() for line in converge_logger.get_lines() if line.startswith(f"B")]

        answersA = [ans if ans != "-9999521" else None for ans in answersA]
        answersB = [ans if ans != "-9999521" else None for ans in answersB]
        answersA = [round(float(ans),2) for ans in answersA if ans != "None"]
        answersB = [round(float(ans)) for ans in answersB if ans != "None"]
        
        # Count how often each final answer appears in its trajectory.
        countA = answersA.count(str(answerA))
        countB = answersB.count(str(answerB))
        
        if countA > countB:
            debate_logger.log(f"Choosing Team A's answer {answerA} with {countA} consistent occurrences over Team B's {answerB} with {countB} occurrences.")
            print(f"Choosing Team A's answer {answerA} with {countA} consistent occurrences over Team B's {answerB} with {countB} occurrences.")
            return answerA
        elif countB >= countA:
            debate_logger.log(f"Choosing Team B's answer {answerB} with {countB} consistent occurrences over Team A's {answerA} with {countA} occurrences.")
            print(f"Choosing Team B's answer {answerB} with {countB} consistent occurrences over Team A's {answerA} with {countA} occurrences.")
            return answerB
        
            
def clean_datasets_test(dataset_name, num, start, mode, base_model, start_stage, temperature, confident_mode, revision_history, coding_model, prompt_version):
    print(f"Cleaning and testing dataset: {dataset_name}, mode: {mode}, base_model: {base_model}, start_stage: {start_stage}, temperature: {temperature}, confident_mode: {confident_mode}, revision_history: {revision_history}, coding_model: {coding_model}, prompt_version: {prompt_version}")
    
    if coding_model is not None:
        assert mode == "debate" and start_stage>=1, "coding_model is only applicable in debate mode with start_stage >= 1"
    
    if mode not in ["single", "debate"]:
        raise ValueError("Mode must be 'single' or 'debate'")
    if mode == "single":
        temperature = temperature[0]
        base_model = base_model[0]
        assert isinstance(temperature, float), "Temperature must be a float value for single mode"
        # assert base_model in ["gpt-4o", "gemini-2.5-pro", "deepseek-reasoner", "deepseek-v3"], "Base model must be one of ['gpt-4o', 'gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v3']"
    elif mode == "debate":
        if start_stage == 0:
            assert isinstance(temperature, list) and len(temperature) == 2, "Temperature must be a list of two float values for debate mode"
        assert confident_mode in ["least_change", "fixed_A", "fixed_B"], "confident_mode must be one of ['least_change', 'fixed_A', 'fixed_B'], None is not allowed in debate mode"
        assert isinstance(base_model, list) and len(base_model) == 2, "Base model must be a list of two models for debate mode"
        assert base_model[0] in ["gpt-4o", "gemini-2.5-pro", "deepseek-reasoner", "deepseek-v3"], "Base model A must be one of ['gpt-4o', 'gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v3']"
        assert base_model[1] in ["gpt-4o", "gemini-2.5-pro", "deepseek-reasoner", "deepseek-v3"], "Base model B must be one of ['gpt-4o', 'gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v3']"
    if revision_history:
        assert mode == "debate", "Revision history is only applicable in debate mode"
    
    count = 0
    correct = 0
    data = get_desc_and_answer(dataset_name)
    for idx, (desc, ans) in enumerate(data):
        if idx < start:
            continue
        if count >= num:
            break
        print(f"Processing problem {idx}")
        count += 1
        
        if mode == "single":
            problem_path = os.path.join(args.output_dir, f"{base_model}_temperature_{str(temperature)}", dataset_name, f"problem_{idx}")
            os.makedirs(problem_path, exist_ok=True)
            try:
                output = get_answer_from_description_single(desc, problem_path, model=base_model, temperature=temperature, start_stage=start_stage)
                answer = get_answer_from_output(output)
            except Exception as e:
                print(f"Error processing problem {idx}: {e}")
                output = traceback.format_exc()
                answer = None
                
        elif mode == "debate":
            # Create separate folders for teams A and B.
            base_modelA = base_model[0]
            base_modelB = base_model[1]
            problem_path = f"history_{mode}_revision_{str(revision_history)}_v{str(prompt_version)}/{base_modelA}_{temperature[0]}_vs_{base_modelB}_{temperature[1]}/{dataset_name}/problem_{idx}"
            os.makedirs(problem_path, exist_ok=True)
            teamA_path = os.path.join(problem_path, "team_A")
            teamB_path = os.path.join(problem_path, "team_B")
            os.makedirs(teamA_path, exist_ok=True)
            os.makedirs(teamB_path, exist_ok=True)
            try:
                if start_stage == 0:
                    answer = get_answer_from_description_debate(desc, problem_path, temperatureA=temperature[0], temperatureB=temperature[1], start_stage=start_stage, confident_mode=confident_mode, revision_history=revision_history, modelA=base_modelA, modelB=base_modelB,prompt_version=prompt_version)
                elif start_stage >=1:
                    modela_path = f"history_single/{base_modelA}_temperature_{temperature[0]}/{dataset_name}/problem_{idx}"
                    modelb_path = f"history_single/{base_modelB}_temperature_{temperature[1]}/{dataset_name}/problem_{idx}"
                    answer = get_answer_from_description_debate(desc=desc, problem_path=problem_path, start_stage=start_stage,modelA_problem_path=modela_path, modelB_problem_path=modelb_path, confident_mode=confident_mode, revision_history=revision_history, modelA=base_modelA, modelB=base_modelB, coding_model=coding_model, prompt_version=prompt_version)
                with open(os.path.join(problem_path, "output_solution.txt"), "w") as f:
                    f.write(str(answer))
            except Exception as e:
                print(f"Error processing problem {idx} in debate mode: {e}")
                output = traceback.format_exc()
                with open(os.path.join(problem_path, "output.txt"), "w") as f:
                    f.write(str(output))
        
        if converge(answer, ans):
            print(f"Problem {idx} solved correctly: {answer}")
            correct += 1
        else:
            print(f"Problem {idx} incorrect: got {answer}, expected {ans}")
        print("Current accuracy: {:.2f}%".format(correct / count * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean datasets and test the model.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to clean and test.")
    parser.add_argument("--num", type=int, default=10, help="Number of problems to process from the dataset.")
    parser.add_argument("--start", type=int, default=0, help="Starting index for processing problems.")
    parser.add_argument("--mode", type=str, choices=["single", "debate"], default="single", help="Mode of operation: single or debate.")
    parser.add_argument("--base_model", nargs="+", default=["gpt-4o"])
    parser.add_argument("--start_stage", type=int, default=0)
    parser.add_argument("--temperature", nargs="+", type=float, default=[0.01])
    parser.add_argument("--confident_mode", type=str, choices=[None, "least_change", "fixed_A", "fixed_B"], default=None, help="Confidence mode for debate: least_change, fixed_A, or fixed_B.")
    parser.add_argument("--revision_history", type=bool, default=False, help="Whether to add revision history in debate mode for agents to make decisions.")
    parser.add_argument("--coding_model", default=None)
    parser.add_argument("--prompt_version", type=int, default=0, help="Prompt version to use, 0,1,2,3")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the results.")
    
    args = parser.parse_args()
    

    clean_datasets_test(dataset_name = args.dataset, num = args.num, start = args.start, mode = args.mode, base_model = args.base_model, start_stage=args.start_stage, temperature=args.temperature, confident_mode=args.confident_mode, revision_history=args.revision_history, coding_model=args.coding_model, prompt_version = args.prompt_version)
    
    
