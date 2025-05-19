from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client(api_key='')
prompt= f"""
Generate a high-quality, visually appealing photograph of a scene or subject directly related to the following narrative context, as if captured by DD.

Narrative Context: "At 10:32 AM (Local time for New York), Isabella Rossi pushes back her chair, a subtle scrape against the floor, and heads out of her office building. The Bleecker Street cafe is about a fifteen-minute walk, a pleasant enough stroll on this slightly overcast Monday morning. She joins the steady stream of pedestrians navigating the sidewalks, a mix of hurried professionals and more leisurely shoppers. The faint aroma of street food mingles with exhaust fumes, the quintessential scent of the city."

Instructions for the Image:
The image should feature:
-   A clear subject directly related to the Narrative Context.
-   Lighting, composition, and focus that give it the aesthetic of a professional, high-engagement social media photograph (like those popular on Instagram or Twitter).
-   Details that align with the World Mood.
-   A composition that is balanced and aesthetically pleasing.

Style:
-   The overall aesthetic should be similar to a high-quality photograph suitable for a popular social media feed, emphasizing photographic quality, clarity, and visual appeal.
-   The image should look as if it were taken by DD from their perspective (e.g., first-person view, a photo of something they are looking at, or a selfie if appropriate to the narrative).
-   Consider an aspect ratio common on social media, such as 1:1 (square) or 4:5 (portrait), if appropriate.

Crucial Exclusions:
-   **The image itself must NOT contain any digital overlays, app interfaces, Instagram/Twitter frames, borders, like buttons, comment icons, usernames, text captions, or any other UI elements.**
-   **No watermarks or logos should be embedded in the image.**
-   The output should be the pure photographic image of the subject as described.

Generate this image."""

response = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt=prompt,
    config=types.GenerateImagesConfig(
        number_of_images= 4,
    )
)
for generated_image in response.generated_images:
  image = Image.open(BytesIO(generated_image.image.image_bytes))
  image.show()