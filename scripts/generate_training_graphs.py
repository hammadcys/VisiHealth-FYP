import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

# Style settings
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Find the tfevents file
event_files = glob.glob('events.out.tfevents.*')
if not event_files:
    print("Error: No tfevents file found in the current directory.")
    exit(1)

event_file = event_files[0]
print(f"Reading events from: {event_file}")

# Load the event accumulator
ea = EventAccumulator(event_file)
ea.Reload()

# Extract available tags
tags = ea.Tags()['scalars']
print(f"Found scalar tags: {tags}")

def extract_data(tag):
    if tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    return [], []

# Extract metrics
train_loss_steps, train_loss_vals = extract_data('Train/Loss')
val_loss_steps, val_loss_vals = extract_data('Val/Loss')
train_acc_steps, train_acc_vals = extract_data('Train/Accuracy')
val_acc_steps, val_acc_vals = extract_data('Val/Accuracy')

# Ensure results directory exists
os.makedirs('results/graphs', exist_ok=True)

# 1. Loss Graph
plt.figure(figsize=(10, 6))
if train_loss_vals:
    plt.plot(train_loss_steps, train_loss_vals, label='Train Loss', color='#1f77b4', linewidth=2)
if val_loss_vals:
    plt.plot(val_loss_steps, val_loss_vals, label='Validation Loss', color='#ff7f0e', linewidth=2)

plt.title('Training and Validation Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('results/graphs/loss_graph.png', dpi=300)
print("Saved: results/graphs/loss_graph.png")
plt.close()

# 2. Accuracy Graph
plt.figure(figsize=(10, 6))
if train_acc_vals:
    plt.plot(train_acc_steps, train_acc_vals, label='Train Accuracy', color='#2ca02c', linewidth=2)
if val_acc_vals:
    plt.plot(val_acc_steps, val_acc_vals, label='Validation Accuracy', color='#d62728', linewidth=2)

plt.title('Training and Validation Accuracy', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('results/graphs/accuracy_graph.png', dpi=300)
print("Saved: results/graphs/accuracy_graph.png")
plt.close()

print("\nSuccessfully generated Accuracy and Loss graphs!")
