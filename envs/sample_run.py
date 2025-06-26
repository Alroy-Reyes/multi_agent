import numpy as np
import numpy as np
from timetabling_env import TimetablingEnv

class TimetablingEnvWithDisplay(TimetablingEnv):
    def render_schedule(self):
        """Display the current room schedule in a tabular format."""
        
        print("\nRoom Assignment Schedule:")
        
        # Column headers (Timeslots)
        print(f"{'Room/Timeslot':<15}", end="")
        for timeslot in range(self.num_timeslots):
            print(f"TS{timeslot + 1:<5}", end="")  # Format each timeslot with consistent spacing
        print()  # Newline after the header
        
        print("-" * (15 + self.num_timeslots * 7))  # Create a separator line for better readability
        
        # Display each room's booking for each timeslot
        for bldg_id, room_types in self.buildings_room_info.items():
            for room_idx, room_type in enumerate(room_types):
                print(f"Room {room_idx + 1} ({room_type:<7}) | ", end="")
                # Display bookings for this room across all timeslots
                for timeslot in range(self.num_timeslots):
                    subject = self.buildings_room_schedule[bldg_id][room_idx][timeslot]
                    if subject == -1:
                        print(f"{'Available':<8}", end="")  # Ensure consistent width for "Available"
                    else:
                        print(f"S{subject + 1:<7}", end="")  # Show subject number with consistent spacing
                print()  # Newline after each room's schedule

# Create a new environment instance
env = TimetablingEnvWithDisplay()

# Sample run of the environment
env.reset()

# Display the schedule for each step
for step in range(10):
    print(f"\nStep {step + 1}:")
    
    # Render the current schedule for better visualization
    env.render_schedule()
    
    # Iterate over agents and take actions
    for agent in env.agents:
        observation = env.observe(agent)
        print(f"Observation for {agent}: {observation}")
        action = np.random.randint(0, env.action_spaces[agent].n)
        print(f"Action taken by {agent}: {action}")
        env.step(action)
        print(f"Reward for {agent}: {env.rewards[agent]}")

    print("-" * 30)

# Close the environment after the run
env.close()
