#!/bin/bash

# Copies example audio files for local demo use.
# Run this only after downloading the Fluent Speech Commands dataset.

set -e

DATASET_DIR="fluent_speech_commands_dataset"
EXAMPLES_DIR="examples"

echo "🔍 Checking for FSC dataset..."

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Error: Dataset directory '$DATASET_DIR' not found."
    echo "Please download the Fluent Speech Commands dataset and extract it to:"
    echo "  fluent_speech_commands_dataset/ in the project root."
    exit 1
fi

# Define source paths
SPEAKER_DIR="$DATASET_DIR/wavs/speakers/4BrX8aDqK2cLZRYl"
FILES=(
    "e1501430-452c-11e9-b1e4-e5985dca719e.wav"
    "65c428f0-452d-11e9-b1e4-e5985dca719e.wav"
    "0fa22ed0-452e-11e9-b1e4-e5985dca719e.wav"
)

# Check if all files exist
echo "🔍 Checking required audio files..."
for file in "${FILES[@]}"; do
    if [ ! -f "$SPEAKER_DIR/$file" ]; then
        echo "❌ Missing file: $SPEAKER_DIR/$file"
        echo "Make sure the dataset is fully extracted."
        exit 1
    fi
done

mkdir -p "$EXAMPLES_DIR"

echo "📎 Copying example files to '$EXAMPLES_DIR/'..."

cp "$SPEAKER_DIR/e1501430-452c-11e9-b1e4-e5985dca719e.wav" "$EXAMPLES_DIR/turn-the-lights-on_activate-lights-none.wav"
cp "$SPEAKER_DIR/65c428f0-452d-11e9-b1e4-e5985dca719e.wav" "$EXAMPLES_DIR/turn-the-lights-on-in-the-kitchen_activate-lights-kitchen.wav"
cp "$SPEAKER_DIR/0fa22ed0-452e-11e9-b1e4-e5985dca719e.wav" "$EXAMPLES_DIR/i-cant-hear-that_increase-volume-none.wav"

echo "✅ Success! Example files copied:"
echo "💡 You can now run your Gradio app locally!"
