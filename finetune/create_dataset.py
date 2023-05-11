
import os
import json


def main(
  file_path: str = "data/input/MSA_Juniper_IBM.txt",
  output_path: str = "data/datasets/merge.jsonl"
):
  with open(file_path, "r") as f:
    text = f.read()
  lines = text.split("\n")
  min_length = 700
  max_overlap = 400
  chunks = []
  for i, line in enumerate(lines):
    if len(chunks) == 0 or len(chunks[-1]) > min_length:
      overlapp = ""
      if len(chunks) > 0:
        overlapp = chunks[-1][-max_overlap:]
        # remove text before first space
        overlapp = overlapp[overlapp.find(" "):]   
      chunks.append(overlapp + "\n" +  line)
    else:
      chunks[-1] += "\n" + line
  print(f"Found {len(chunks)} chunks")

  # write instruction json
  instructions = [
    {
      "instruction": "Write a passage of a financial contract that answers the user's question",
      "input": "",
      "output": chunk
    } for chunk in chunks
  ]
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  with open(output_path, "w") as f:
    for instruction in instructions:
        f.write(json.dumps(instruction) + "\n")


if __name__ == "__main__":
  from jsonargparse import CLI

  CLI(main)
  