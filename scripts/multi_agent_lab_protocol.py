##############################################################################
import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
import requests
from xml.etree import ElementTree as ET
import re
from collections import defaultdict
from typing import Union
from datetime import datetime
##############################################################################

class DispensedPlate(BaseModel):
    """A plate that has been dispensed with bacteria and phages and serially diluted"""
    state : str = Field(description="The state of the experiment")

class PreparedMicroscope(BaseModel):
    """A microscope that is ready to receive the dispensed plate"""
    state : str = Field(description="The state of the experiment")

class DispensedPlateInScope(BaseModel):
    """A dispensed plate that is in the microscope"""
    state : str = Field(description="The state of the experiment")

class WellPlateDuringImaging(BaseModel):
    """A well plate that is being imaged"""
    state : str = Field(description="The state of the experiment")

class FinishedPlateImaging(BaseModel):
    """A well plate that has finished imaging"""
    state : str = Field(description="The state of the experiment")

class WellPlateDoneInScope(BaseModel):
    """A well plate that is done in the microscope"""
    state : str = Field(description="The state of the experiment")

class ScopeReadyForArm(BaseModel):
    """A microscope with a finished well plate that is ready to be moved by the arm"""
    state : str = Field(description="The state of the experiment")

class ImagedPlateInOpentron(BaseModel):
    """A well plate that is in the Opentron"""
    state : str = Field(description="The state of the experiment")

class DispensedAgarPlate(BaseModel):
    """A well plate that has been dispensed with agar"""
    state : str = Field(description="The state of the experiment")

class AgarPlateToIncubate(BaseModel):
    """An agar plate that is ready to be incubated"""
    state : str = Field(description="The state of the experiment")

class IncubatedAgarPlate(BaseModel):
    """An agar plate that has been incubated"""
    state : str = Field(description="The state of the experiment")

class AgarPlateInScope(BaseModel):
    """An agar plate that is in the microscope"""
    state : str = Field(description="The state of the experiment")

class AgarPlateDuringImaging(BaseModel):
    """An agar plate that is being imaged"""
    state : str = Field(description="The state of the experiment")

class FinishedAgarImaging(BaseModel):
    """An agar plate that has finished imaging"""
    state : str = Field(description="The state of the experiment")
    


def update_state(old_state, add_state):
    return f"{old_state}\n- {add_state}"

async def initial_dispense_2(start_string : str, role: Role = None) -> DispensedPlate:
    """Dispenses the bacteria and phages into the well plate"""
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"pipetting_robot_agent: Dispensing the bacteria and phages into the well plate ({current_time})"
    with open("protocol_run.md", "w") as f:
        f.write(f"#### {s}\n")
    dispensed_plate = DispensedPlate(state = update_state("", s))
    return dispensed_plate

async def microscope_prep_3(dispensed_plate : DispensedPlate, role : Role = None) -> PreparedMicroscope:
    """Prepares the microscope for receiving the dispensed plate"""
    old_state = dispensed_plate.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"microscope_agent: Preparing the microscope for receiving the dispensed plate ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    prepared_microscope = PreparedMicroscope(state = update_state(old_state, s))
    return prepared_microscope

async def move_opentron_to_microscope_4(prepared_microscope : PreparedMicroscope, role : Role = None) -> DispensedPlateInScope:
    """Moves the Opentron to the microscope"""
    old_state = prepared_microscope.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"arm_agent: Moving the Opentron to the microscope ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    dispensed_plate_in_scope = DispensedPlateInScope(state = update_state(old_state, s))
    return dispensed_plate_in_scope

async def start_imaging_5(dispensed_plate_in_scope : DispensedPlateInScope, role : Role = None) -> WellPlateDuringImaging:
    """Starts the imaging process"""
    old_state = dispensed_plate_in_scope.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"microscope_agent: Starting the imaging process ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    well_plate_during_imaging = WellPlateDuringImaging(state = update_state(old_state, s))
    return well_plate_during_imaging

async def run_plate_analysis_6(well_plate_during_imaging : WellPlateDuringImaging, role : Role = None) ->FinishedPlateImaging:
    """Runs the plate analysis"""
    old_state = well_plate_during_imaging.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"server_agent: Running the plate analysis ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    finished_plate_imaging = FinishedPlateImaging(state = update_state(old_state, s))
    return finished_plate_imaging

async def end_imaging_7(finished_plate_imaging : FinishedPlateImaging, role : Role = None) -> WellPlateDoneInScope:
    """Ends the imaging process"""
    old_state = finished_plate_imaging.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"server_agent: Ending the imaging process ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    well_plate_done_in_scope = WellPlateDoneInScope(state = update_state(old_state, s))
    return well_plate_done_in_scope

async def microscope_finish_8(well_plate_done_in_scope : WellPlateDoneInScope, role : Role = None) -> ScopeReadyForArm:
    """Finishes the microscope imaging process and prepares the microscope for the arm"""
    old_state = well_plate_done_in_scope.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"microscope_agent: Finishing the microscope imaging process and preparing the microscope for the arm ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    scope_ready_for_arm = ScopeReadyForArm(state = update_state(old_state, s))
    return scope_ready_for_arm

async def move_to_opentron_9(scope_ready_for_arm : ScopeReadyForArm, role : Role = None) -> ImagedPlateInOpentron:
    """Moves the microscope to the Opentron"""
    old_state = scope_ready_for_arm.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"arm_agent: Moving the microscope to the Opentron ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    imaged_plate_in_opentron = ImagedPlateInOpentron(state = update_state(old_state, s))
    return imaged_plate_in_opentron

async def dispense_agar_10(imaged_plate_in_opentron : ImagedPlateInOpentron, role : Role = None) -> DispensedAgarPlate:
    """Dispenses agar"""
    old_state = imaged_plate_in_opentron.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"pipetting_robot_agent: Dispensing agar ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    dispensed_agar_plate = DispensedAgarPlate(state = update_state(old_state, s))
    return dispensed_agar_plate

async def move_opentron_to_incubator_11(dispensed_agar_plate : DispensedAgarPlate, role : Role = None) -> AgarPlateToIncubate:
    """Moves the Opentron to the incubator"""
    old_state = dispensed_agar_plate.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"arm_agent: Moving agar plate from the Opentron to the incubator ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    agar_plate_to_incubate = AgarPlateToIncubate(state = update_state(old_state, s))
    return agar_plate_to_incubate

async def incubate_agar_12(agar_plate_to_incubate : AgarPlateToIncubate, role : Role = None) -> IncubatedAgarPlate:
    """Incubates the agar"""
    old_state = agar_plate_to_incubate.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"incubator_agent: Incubating the agar plate ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    incubated_agar_plate = IncubatedAgarPlate(state = update_state(old_state, s))
    return incubated_agar_plate

async def move_incubator_to_microscope_13(incubated_agar_plate : IncubatedAgarPlate, role : Role = None) -> AgarPlateInScope:
    """Moves the incubator to the microscope"""
    old_state = incubated_agar_plate.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"arm_agent: Moving the incubator to the microscope ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    agar_plate_in_scope = AgarPlateInScope(state = update_state(old_state, s))
    return agar_plate_in_scope

async def start_agar_imaging_14(agar_plate_in_scope : AgarPlateInScope, role : Role = None) -> AgarPlateDuringImaging:
    """Starts the agar imaging process"""
    old_state = agar_plate_in_scope.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"microscope_agent: Starting the agar imaging process ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    agar_plate_during_imaging = AgarPlateDuringImaging(state = update_state(old_state, s))
    return agar_plate_during_imaging

async def run_agar_analysis_15(agar_plate_during_imaging : AgarPlateDuringImaging, role : Role = None) -> FinishedAgarImaging:
    """Runs the agar analysis"""
    old_state = agar_plate_during_imaging.state
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"server_agent: Running the agar analysis ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    finished_agar_imaging = FinishedAgarImaging(state = update_state(old_state, s))
    current_time = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
    s = f"server_agent: Finished the agar analysis ({current_time})"
    with open("protocol_run.md", "a") as f:
        f.write(f"#### {s}")
        f.write(f"{old_state}\n")

    return finished_agar_imaging
    
    

# Main function
async def main():
    agents = []
    pipetting_robot_agent = Role(name="Pipetting Robot",
                            profile="An agent that prepares the agar plates for the experiment.",
                            goal="To prepare the agar plates for the experiment.",
                            constraints=None,
                            actions=[initial_dispense_2, dispense_agar_10])
    agents.append(pipetting_robot_agent)

    microscope_agent = Role(name="Microscope",
                            profile="An agent that analyzes the well plates and agar plates.",
                            goal="To analyze the well plates and agar plates.",
                            constraints=None,
                            actions=[microscope_prep_3,
                                     start_imaging_5,
                                     microscope_finish_8,
                                     start_agar_imaging_14])
    agents.append(microscope_agent)

    arm_agent = Role(name="Arm",
                    profile="A robotic arm agent that transfers the plates between the pipetting robot, incubator, and microscope.",
                    goal="To transfer the agar plates from the pipetting robot to the microscope.",
                    constraints=None,
                    actions=[move_opentron_to_microscope_4, 
                             move_to_opentron_9, 
                             move_opentron_to_incubator_11,
                             move_incubator_to_microscope_13])
    agents.append(arm_agent)

    sever_agent = Role(name="Server", 
                    profile="An agent that analyzes the data and suggests follow up studies.", 
                    goal="To analyze the data and suggest follow up studies.", 
                    constraints=None, 
                    actions=[run_plate_analysis_6,
                             end_imaging_7,
                             run_agar_analysis_15])
    agents.append(sever_agent)

    incubator_agent = Role(name="Incubator",
                        profile="An agent that incubates the agar plates.",
                        goal="To incubate the agar plates.",
                        constraints=None,
                        actions=[incubate_agar_12])
    agents.append(incubator_agent)
    

    team = Team(name="Protocol runner", profile="A team specialized in running the given protocol", investment=0.7)

    team.hire(agents)
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    user_request = "Start the protocol!"
    responses = await team.handle(Message(content=user_request, role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()