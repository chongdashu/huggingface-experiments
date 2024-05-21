# Based on the blogpost:
# https://medium.com/@kyeg/get-started-with-paligemma-locally-on-cloud-the-all-new-multi-modal-model-from-google-f88a97b9ead6
#
# !pip install torch transformers pillow requests swarms==4.9.7

from libs.paligemma import PaliGemma


def main() -> None:
    # Initialize the PaliGemma model
    model = PaliGemma()

    # Define the text prompt and image URL
    text_prompt = "A beautiful sunset over the ocean."
    image_url = "https://example.com/sunset.jpg"

    # Run the PaliGemma model
    output = model.run(text_prompt, image_url)
    # Print the generated output
    print(output)


if __name__ == "__main__":
    main()
