import re
import csv
import os

# Open the text file
with open("resultmoex1.txt", "r") as file:
    lines = file.readlines()

# Prepare output data
data = []

for line in lines:
    # Match Epoch, Test Loss, and Overall Accuracy
    match = re.search(r"Epoch \[(\d+)/\d+\], Test Loss: ([\d.]+), Overall Accuracy: ([\d.]+)%", line)
    if match:
        epoch = int(match.group(1))
        test_loss = float(match.group(2))
        accuracy = float(match.group(3))
        data.append([epoch, test_loss, accuracy])

# Customize export location
export_location = "C:/Users/sherl/Downloads/resultmoex1.csv"  # Replace with your desired path
os.makedirs(os.path.dirname(export_location), exist_ok=True)  # Ensure the directory exists

# Write to CSV file
with open(export_location, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Epoch", "Test Loss", "Overall Accuracy"])
    writer.writerows(data)

print(f"Data has been extracted to {export_location}")
