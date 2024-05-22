# Based on the blogpost:
# https://medium.com/@kyeg/get-started-with-paligemma-locally-on-cloud-the-all-new-multi-modal-model-from-google-f88a97b9ead6
#
# !pip install torch transformers pillow requests swarms==4.9.7

from libs.paligemma import PaliGemma


def main() -> None:
    # Initialize the PaliGemma model
    force_download = False
    model = PaliGemma(force_download=force_download)

    # Define the text prompt and image URL
    text_prompt = "You are an eloquent poet. Describe what you see in the following image in an illustrative and compelling manner."
    image_url = "https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/8b21ba78250f852ca5990063866b1ace6432521d0251bde7f8de783b22c99a6d?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27bee.jpg%3B+filename%3D%22bee.jpg%22%3B&response-content-type=image%2Fjpeg&Expires=1716554277&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxNjU1NDI3N319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy84YjIxYmE3ODI1MGY4NTJjYTU5OTAwNjM4NjZiMWFjZTY0MzI1MjFkMDI1MWJkZTdmOGRlNzgzYjIyYzk5YTZkP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=rNo%7EgrdHn%7EGa6kD-Xx4YPFtoC6u1BfiXtvbbb4hESN6uuu1aQuHsLQEtfc8vNJ9K4m8o3JBPpyUE9xSfah7sFw1EEHmRmIYFjFE3IgFOI%7EpRAT2B5FulFhfmSLzBztgTLVgtAvmOQjLinSGRSZ442XSNI5BHsnOmAqlQ4lbb5OUH3bfPJtd64--rTpKIVqNKUwJoEld9Sf29SSezgOYPeiHoaP5Ds1nhXPcmsupzpvv58JjJsaNhlKlyUWbwFwaCmf2csNu7v0yBJTJ06IXkqcH0dpGXv%7EzqIdqx6jeIaHJulG0uKIhaCXTalmToE3hkwtwdacOid-%7Ed5SL82rcwCg__&Key-Pair-Id=KVTP0A1DKRTAX"

    # Run the PaliGemma model
    output = model.run(text_prompt, image_url)

    # Print the generated output
    print(output)


if __name__ == "__main__":
    main()
