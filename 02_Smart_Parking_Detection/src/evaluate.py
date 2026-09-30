import os

# Results from the parking occupancy detector
total_spaces = 8
free_spaces = 4
occupied_spaces = 4

# Basic occupancy statistics
occupied_percentage = (occupied_spaces / total_spaces) * 100
free_percentage = (free_spaces / total_spaces) * 100

output_dir = "02_Smart_Parking_Detection/evaluation_results"
os.makedirs(output_dir, exist_ok=True)

result_file = os.path.join(output_dir, "evaluation.txt")

with open(result_file, "w") as f:
    f.write("SMART PARKING SPACE DETECTION - EVALUATION\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total parking spaces : {total_spaces}\n")
    f.write(f"Free spaces          : {free_spaces}\n")
    f.write(f"Occupied spaces      : {occupied_spaces}\n")
    f.write(f"Free space percentage: {free_percentage:.2f}%\n")
    f.write(f"Occupied percentage  : {occupied_percentage:.2f}%\n")
    f.write("Video frames processed: 844\n")
    f.write("=" * 50 + "\n")

print("=" * 50)
print("SMART PARKING EVALUATION")
print("=" * 50)
print(f"Total spaces : {total_spaces}")
print(f"Free spaces  : {free_spaces}")
print(f"Occupied     : {occupied_spaces}")
print(f"Free         : {free_percentage:.2f}%")
print(f"Occupied     : {occupied_percentage:.2f}%")
print(f"Saved to     : {result_file}")
print("=" * 50)