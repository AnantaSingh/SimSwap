import os
import random
import subprocess

def run_test_one_image(pic_a_path, pic_b_path, output_path):
    cmd = [
        "python", "test_one_image.py",
        "--crop_size", "224",
        "--use_mask",
        "--name", "people",
        "--Arc_path", "arcface_model/arcface_checkpoint.tar",
        "--pic_a_path", pic_a_path,
        "--pic_b_path", pic_b_path,
        "--output_path", output_path,
        "--no_simswaplogo"
    ]
    subprocess.run(cmd, check=True)

def get_valid_image(celeb_dir, max_attempts=10):
    files = [f for f in os.listdir(celeb_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)
    for filename in files[:max_attempts]:
        img_path = os.path.join(celeb_dir, filename)
        if os.path.isfile(img_path):
            return img_path
    return None

def generate_multiple_deepfakes(celeb1_dir, celeb2_dir, output_dir, pair_name, num_deepfakes=10):
    celeb1_name = os.path.basename(celeb1_dir)
    celeb2_name = os.path.basename(celeb2_dir)
    celeb1_output_dir = os.path.join(output_dir, celeb1_name)
    celeb2_output_dir = os.path.join(output_dir, celeb2_name)
    os.makedirs(celeb1_output_dir, exist_ok=True)
    os.makedirs(celeb2_output_dir, exist_ok=True)

    successful_deepfakes = 0
    attempts = 0
    max_attempts = num_deepfakes * 2

    while successful_deepfakes < num_deepfakes and attempts < max_attempts:
        source_path1 = get_valid_image(celeb1_dir)
        target_path2 = get_valid_image(celeb2_dir)
        source_path2 = get_valid_image(celeb2_dir)
        target_path1 = get_valid_image(celeb1_dir)

        if not all([source_path1, target_path2, source_path2, target_path1]):
            print(f"Attempt {attempts + 1}: Failed to find valid images")
            attempts += 1
            continue

        output_path1 = os.path.join(celeb2_output_dir, f"{pair_name}_{os.path.basename(source_path1)}_to_{os.path.basename(target_path2)}")
        output_path2 = os.path.join(celeb1_output_dir, f"{pair_name}_{os.path.basename(source_path2)}_to_{os.path.basename(target_path1)}")

        try:
            run_test_one_image(source_path1, target_path2, output_path1)
            run_test_one_image(source_path2, target_path1, output_path2)
            successful_deepfakes += 1
        except Exception as e:
            print(f"Error generating deepfake: {e}")

        attempts += 1

    print(f"Generated {successful_deepfakes} deepfakes for {pair_name}")

if __name__ == '__main__':
    # Set up directories
    celebrities_dir = "crop_224/celebrities"
    output_dir = "output/deepfakes"

    # Generate deepfakes for Leonardo and Orlando
    leo_dir = os.path.join(celebrities_dir, "Leonardo_DiCaprio")
    orlando_dir = os.path.join(celebrities_dir, "Orlando_Bloom")
    generate_multiple_deepfakes(leo_dir, orlando_dir, output_dir, "leo_orlando", num_deepfakes=10)

    # Generate deepfakes for Lindsay and Tom
    lindsay_dir = os.path.join(celebrities_dir, "Lindsay_Lohan")
    tom_dir = os.path.join(celebrities_dir, "Tom_Cruise")
    generate_multiple_deepfakes(lindsay_dir, tom_dir, output_dir, "lindsay_tom", num_deepfakes=10)

    # Test swap for brad and leo
    run_test_one_image("crop_224/brad.jpg", "crop_224/leo.jpg", "output/")
    run_test_one_image("crop_224/leo.jpg", "crop_224/brad.jpg", "output/")