#!/usr/bin/python
# -*- coding: utf-8 -*-

import PyPDF2
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from streaming import stream
from openai import OpenAI
from pydub import AudioSegment
import math
import os

api_key = os.getenv("OPENAI_API_KEY")

pdfFileObj = open("""C:/Users/user/file""", "rb") ## provide proper filename
pdfReader = PyPDF2.PdfReader(pdfFileObj)

pages = []
i = 1

chat = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4-1106-preview", temperature=0)
client = OpenAI(api_key=api_key)

for page in pdfReader.pages[7:9]:
    page = page.extract_text()

    user = HumanMessage(content=f"""Tekst do przetłumaczenia: {page}""")
    system = SystemMessage(content="""Jestem specjalistą w tłumaczeniach z angielskiego na polski.
               Zasady:
               - Wzory podaję w formacie Latex,
               - Usuwam kropki po końcu KAŻDEGO zdania i zamiast nich stawiam znak "#$#" (TO JEST BARDZO WAŻNE!!! NA PODSTAWIE TEGO, ZDANIA BĘDĄ PÓŹNIEJ DZIELONE),
               - Nie muszę tłumaczyć jeden do jednego, mogę wprowadzać zmiany o ile nie zmienia to sensu wypowiedzi,
               ### Przykład:
               Użytkownik: "Tekst do przetłumaczenia: My name is Mark. I live in England. I like football.
               Ja: "Mam na imię Mark #$# Mieszkam w Anglii #$# Lubię piłkę nożną"
               ###""")

    conversation: list[HumanMessage | SystemMessage | AIMessage] = [system, user]

    response = stream(chat, conversation)
    page = response.split("#$#")
    length = len(response)
    max_length_of_txt_to_speech_model = 4096 - 3500
    how_many_time_length_exceeded = math.ceil(length / max_length_of_txt_to_speech_model)
    if how_many_time_length_exceeded > 1:
        total_length = 0
        single_page = []
        for sentence in page:
            total_length += len(sentence)
            if total_length > max_length_of_txt_to_speech_model:
                pages.append(single_page)
                total_length = len(sentence)
                single_page = [sentence]
            else:
                single_page.append(sentence)
        pages.append(single_page)
    else:
        pages.append(page)

i = 1
for page in pages:
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=f"""{page}""",
    )
    response.stream_to_file(f"""audio_files/output{i}.mp3""")
    i += 1

# List of mp3 files to merge
files_to_merge = [f"audio_files/output{j}.mp3" for j in range(1, i)]

# Loading the first file
combined = AudioSegment.from_file(files_to_merge[0], format="mp3")

# Loop through the rest of the mp3 files and append audio
for mp3_file in files_to_merge[1:]:
    next_audio = AudioSegment.from_file(mp3_file, format="mp3")
    combined += next_audio

# Export the combined audio to a new file
combined.export("result.mp3", format="mp3")


