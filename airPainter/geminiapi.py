#pip install -q -U google-generativeai
import google.generativeai as genai
import PIL.Image
import markdown
import asyncio

MY_API_KEY = "YOUR_API_KEY"

genai.configure(api_key=MY_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')


def printResponse(pic):
    img = PIL.Image.open(pic)

    response = model.generate_content([
                                          "describe the image or text(if on the pic) and if you find any maths related problem then also solve it.(keep the your prompt under 100 words striclty), if the image is a black square, then say nothing just reply with '*'",
                                          img], stream=True)

    response.resolve()

    markdown_text = markdown.markdown(response.text)

    print(markdown_text)
    print(response.prompt_feedback)


