# extracts gauss codes from the katlas dataset

from tqdm import tqdm
import os

GC_IDENTIFIER = "invariant:Gauss_Code"
BREAKPOINT = "> "
LINK_IDENTIFIER = "{"

UNKNOT_ID = "0_1"
UNKNOT_CODE = "-1,1"

source_filename = "katlas.rdf"
output_filename = "katlas_gauss.rdf"

# reset output_file
if os.path.exists(output_filename):
    os.remove(output_filename)

# filter lines
with open(source_filename, "r") as source_file:
    with open(output_filename, "a") as output_file:
        for line in tqdm(source_file.readlines()):
            if GC_IDENTIFIER in line:
                components = line.split(BREAKPOINT)

                # arcane string manipulation
                knot_id = components[0].split(":")[-1]
                code = components[-1][1:-4].replace(" ", "")

                # we don't want to count links
                if LINK_IDENTIFIER in code:
                    continue

                # specially handle the unknot
                if knot_id == UNKNOT_ID:
                    # we can't have an empty code for technical reasons
                    # so instead we use the simplest non-empty code
                    code = UNKNOT_CODE

                new_string = f"{knot_id}:{code}\n"
                output_file.write(new_string)