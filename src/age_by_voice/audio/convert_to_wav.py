import os

from pydub import AudioSegment


def convert_to_wav(input_path, output_folder):
    """
    Converts a single audio file or all audio files in a folder to WAV format.
    Args:
        input_path (str): Path to the audio file or folder containing audio files.
        output_folder (str): Path to the folder where the converted WAV files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.isfile(input_path):
        # Convert single file
        convert_file_to_wav(input_path, output_folder)
    elif os.path.isdir(input_path):
        # Convert all files in the folder
        for file_name in os.listdir(input_path):
            file_path = os.path.join(input_path, file_name)
            if os.path.isfile(file_path):
                convert_file_to_wav(file_path, output_folder)
    else:
        print(f"Invalid input path: {input_path}")


def convert_file_to_wav(file_path, output_folder):
    """
    Converts a single audio file to WAV format.
    Args:
        file_path (str): Path to the audio file.
        output_folder (str): Path to the folder where the converted WAV file will be saved.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_folder, f"{base_name}.wav")
        audio.export(output_path, format="wav")
        print(f"Converted: {file_path} -> {output_path}")
    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")


# Example usage
if __name__ == "__main__":
    input_path = "/Users/macpro/Documents/Programming/GitStuff/Age_by_Voice/data"  # File or folder path
    output_folder = "/Users/macpro/Documents/Programming/GitStuff/Age_by_Voice/output"
    convert_to_wav(input_path, output_folder)
