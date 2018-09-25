import re
from io import StringIO
import pandas as pd
import os.path

def parse_x1(num, header=['Center Number', 'Atomic Number', 'Atomic Type', 'X', 'Y', 'Z']):
    # Extract the Standard Orientation
    with open("data/raw/outs/" + str(num) + ".out", "r") as f:
        outtxt = f.read()

    # Find the final occurrence of "Standard orientation"
    header_expression = re.finditer(r"Standard orientation", outtxt)
    header_expression = list(header_expression)[-1]
    header_end = int(header_expression.end())

    # Find the starting boundary
    block = " ---------------------------------------------------------------------"
    boundary1 = outtxt.find(block, header_end)
    boundary2 = outtxt.find(block, boundary1 + len(block))
    boundary3 = outtxt.find(block, boundary2 + len(block))

    # Extract all the data elements
    values = re.split('\s{1,}|\n', outtxt[boundary2+len(block): boundary3])[1:-1]

    # Group into blocks of 6
    values = [values[a:a+6] for a in range(0,len(values),6)]

    # Create final data frame & remove atomic type column.
    df = pd.DataFrame(values, columns=header)
    df = df.drop(columns=['Atomic Type'])

    return df


def parse_x2(num, header=['Alpha  occ. eigenvalues', 'Alpha  virt. eigenvalues']):
    # Extract the Alpha occ. & virt eigenvalues

    with open("data/raw/outs/" + str(num) + ".out", "r") as f:
        outtxt = f.read()

    # Find the final occurrence of "Alpha  occ. eigenvalues"
    header_expression = re.finditer(r"( The electronic state is (.*).(\n)) (Alpha  occ. eigenvalues)", outtxt)
    header_expression = list(header_expression)[-1]
    header_end = int(header_expression.span(2)[1])

    # Find the starting point for the "Alpha virt. eigenvalues"
    virt_starting_pos = outtxt.find("Alpha virt. eigenvalues", header_end)
    virt_ending_pos = outtxt.find("Condensed to atoms", virt_starting_pos)

    # Prepare the blocks
    occ_block = outtxt[header_end:virt_starting_pos]
    virt_block = outtxt[virt_starting_pos:virt_ending_pos]

    occ_block = map(lambda line: line[line.index("--")+2:] if "--" in line else "", occ_block.split("\r\n"))
    virt_block = map(lambda line: line[line.index("--")+2:] if "--" in line else "", virt_block.split("\r\n"))

    occ_block = " ".join(occ_block)
    virt_block = " ".join(virt_block)

    occ_block = re.split('\s{1,}', occ_block)[1:-1]
    virt_block = re.split('\s{1,}', virt_block)[1:-1]

    # Create final data frame & remove atomic type column.
    df_occ = pd.DataFrame(occ_block, columns=["Alpha  occ. eigenvalues"])
    df_virt = pd.DataFrame(virt_block, columns=["Alpha  virt. eigenvalues"])

    return (df_occ, df_virt)


def parse_x3(num, header=["Atom Number", "Atom"]):
    # Condensed to atoms (all electrons)
    with open("data/raw/outs/" + str(num) + ".out", "r") as f:
        outtxt = f.read()

    # Electric Field Gradient Eigenvalues
    header_expression = re.finditer(r"          Condensed to atoms \(all electrons\):", outtxt)
    
    # Use the second occurrence in the file.
    # Some examples (like 228) only have a single occurrence, so use the final one.
    header_expression = list(header_expression)[-1]

    header_end = int(header_expression.end())

    # Find leading and closing barrier
    leading_pos = outtxt.find("\r\n", header_end)
    closing_pos = outtxt.find(" Mulliken charges:", leading_pos)

    # Split into separate lines
    lines = outtxt[leading_pos+2:closing_pos].split("\n")[1:]

    # Create each dataframe
    dataframes = []
    col_start = 1
    data_line_start = 0
    for idx, line in enumerate(lines):
        if line.startswith("              ") or idx == len(lines)-1:
            # We've reached an ending point
            df = pd.read_csv(StringIO(str("\n".join(lines[data_line_start:idx])).decode('utf-8')), delim_whitespace=True, header=None, names=header + [str(i) for i in range(col_start,col_start+6)])

            # Remove duplicate Atom & Atom Number columns
            del df["Atom"]
            del df["Atom Number"]
            
            dataframes.append(df)

            # Adjust starting column
            col_start +=6

            # Adjust starting block
            data_line_start = idx + 1

    return pd.concat(dataframes, axis= 1)


def parse_x4(num, header=["Atom Number", "Electric Potential", "Val1", "Val2", "Val3"]):
    # Electric Field Gradient, Eigenvalues
    with open("data/raw/outs/" + str(num) + ".out", "r") as f:
        outtxt = f.read()

    # Electric Field Gradient Eigenvalues
    header_expression = re.finditer(r"            Electrostatic Properties Using The SCF Density", outtxt)
    
    # Use the second occurrence in the file.
    # Some examples (like 228) only have a single occurrence, so use the final one.
    header_expression = list(header_expression)[-1]

    header_end = int(header_expression.end())

    # Find leading and closing barrier
    barrier_str = " **********************************************************************\r\n"
    leading_pos = outtxt.find(barrier_str, header_end)
    closing_pos = outtxt.find(" -----------------------------------------------------------------", leading_pos + len(barrier_str))
    return pd.read_csv(StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1]).decode('utf-8')), delim_whitespace=True, header=None, names=header)


def parse_x5(num, header=["Atom Number", "Electric Potential", "X", "Y", "Z"]):
    # Electric Field Gradient, Eigenvalues
    with open("data/raw/outs/" + str(num) + ".out", "r") as f:
        outtxt = f.read()

    # Electric Field Gradient Eigenvalues
    header_expression = re.finditer(r"    Center     Electric         -------- Electric Field --------\r\n               Potential          X             Y             Z", outtxt)
    
    # Use the second occurrence in the file.
    # Some examples (like 228) only have a single occurrence, so use the final one.
    header_expression = list(header_expression)[-1]

    header_end = int(header_expression.end())

    # Find leading and closing barrier
    barrier_str = " -----------------------------------------------------------------"
    leading_pos = outtxt.find(barrier_str, header_end)
    closing_pos = outtxt.find(barrier_str, leading_pos + len(barrier_str))
    return pd.read_csv(StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1]).decode('utf-8')), delim_whitespace=True, header=None, names=header)


def parse_x6(num):
    # Electric Field Gradient, Coordinates
    with open("data/raw/outs/" + str(num) + ".out", "r") as f:
        outtxt = f.read()

    # Electric Field Gradient Eigenvalues
    results = []
    for group in ["XX            YY            ZZ", "XY            XZ            YZ"]:
        header_expression = re.finditer(r"    Center         ---- Electric Field Gradient ----\r\n                     " + group, outtxt)
        
        # Use the second occurrence  in the file.
        # Some examples (like 228) only have a single occurrence, so use the final one.
        header_expression = list(header_expression)[-1]

        # Get the ending position
        header_end = int(header_expression.end())

        # Find leading and closing barrier
        barrier_str = " -----------------------------------------------------"
        leading_pos = outtxt.find(barrier_str, header_end)
        closing_pos = outtxt.find(barrier_str, leading_pos + len(barrier_str))

        # Append to results
        results.append(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1])

    # Prepare Data Frames
    df1 = pd.read_csv(StringIO(str(results[0]).decode('utf-8')), delim_whitespace=True, header=None, names=["Atom Number", "XX", "YY", "ZZ"])
    df2 = pd.read_csv(StringIO(str(results[1]).decode('utf-8')), delim_whitespace=True, header=None, names=["Atom Number", "XY", "XZ", "YZ"])
    del df2["Atom Number"]

    # Return vertically concatenated dataframes
    return pd.concat([df1, df2], axis= 1)


def parse_x7(num, header=["Atom Number", "Eigen 1", "Eigen 2", "Eigen 3"]):
    # Electric Field Gradient, Eigenvalues
    with open("data/raw/outs/" + str(num) + ".out", "r") as f:
        outtxt = f.read()

    # Electric Field Gradient Eigenvalues
    header_expression = re.search(r"    Center         ---- Electric Field Gradient ----\r\n                   ----       Eigenvalues       ----", outtxt)
    header_end = int(header_expression.end())

    # Find leading and closing barrier
    barrier_str = " -----------------------------------------------------"
    leading_pos = outtxt.find(barrier_str, header_end)
    closing_pos = outtxt.find(barrier_str, leading_pos + len(barrier_str))

    return pd.read_csv(StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1]).decode('utf-8')), delim_whitespace=True, header=None, names=header)
       

def parse_x8(num):
    # total SCF density
    with open("data/raw/fchks/Anth_" + str(num) + ".fch", "r") as f:
        outtxt = f.read()

    # Using a regular expression, find the number of data elements.
    header_expression = re.search(r"Total SCF Density                          R   N=(\s)+(\d+)", outtxt)
    num_elems = int(header_expression.group(2))
    header_end = int(header_expression.end())

    # Extract all the data elements
    values = re.split('\s{1,}|\n', outtxt[header_end:])[1:num_elems+1]
    
    # Verify that each value is numeric.
    for val in values:
        try:
            float(val)
        except ValueError:
            raise ValueError('A non-numeric value has been processed.')

    return pd.DataFrame(values)


def parse_x9(num):
    with open("data/raw/fchks/Anth_" + str(num) + ".fch", "r") as f:
        outtxt = f.read()

    # Using a regular expression, find the number of data elements.
    header_expression = re.search(r"Alpha MO coefficients                      R   N=(\s)+(\d+)", outtxt)
    num_elems = int(header_expression.group(2))
    header_end = int(header_expression.end())

    # Extract all the data elements
    values = re.split('\s{1,}|\n', outtxt[header_end:])[1:num_elems+1]

    # Verify that each value is numeric.
    for val in values:
        try:
            float(val)
        except ValueError:
            raise ValueError('A non-numeric value has been processed.')

    return pd.DataFrame(values)


def parse_x10_through_x17(num):
    # Parses through the remaining data in the excel sheet.
    # Return the row for the data number
    # [u'Input', u'Key', u'Associated data', u'X10: Category Method', u'X11: Temperature (K)', u'X12: [Salt*Valency]', u'X13: Category Salt type', u'X14: [Buffer] (mM)', u'X15: pH', u'X16: CI #', u'X17: CI ', u'Unnamed: 11', u'Output: logK']

    dfs = pd.read_excel("data/raw/x10_x17/X10_X17_WAVE2.xlsx", sheet_name=None)['COMPUTER SCIENTISTS LOOK HERE']
    dfs = dfs.loc[dfs[u'Input'] == num]
    return dfs


def create_data_item(num):
    # Uses helper methods to create the data for item of given number.
    dfs_10_17 = parse_x10_through_x17(num)

    data_elements = {
        "x1" :  parse_x1(num),
        "x2_occ" :  parse_x2(num)[0],
        "x2_virt" :  parse_x2(num)[1],
        "x3" :  parse_x3(num) if num != 358 and num != 381 else 0,
        "x4" :  parse_x4(num),
        "x5" :  parse_x5(num),
        "x6" :  parse_x6(num),
        "x7" :  parse_x7(num),
        "x8" :  parse_x8(num) if num != 228 and num != 391 else 0,
        "x9" :  parse_x9(num),
        "x10": dfs_10_17['X10: Category Method'],
        "x11": dfs_10_17['X11: Temperature (K)'],
        "x12": dfs_10_17['X12: [Salt*Valency]'],
        "x13": dfs_10_17['X13: Category Salt type'],
        "x14": dfs_10_17['X14: [Buffer] (mM)'],
        "x15": dfs_10_17['X15: pH'],
        "x16": dfs_10_17['X16: CI #'],
        "x17": dfs_10_17['X17: CI '],
        "output": dfs_10_17['Output: logK']
    }


for num in range(1,1000):
    # Check if file exists
    if os.path.isfile("data/raw/outs/" + str(num) + ".out") and os.path.isfile("data/raw/fchks/Anth_" + str(num) + ".fch"):
        create_data_item(num)