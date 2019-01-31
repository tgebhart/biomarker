import re
from io import StringIO
import os
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# from keras import layers
# from keras import models
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras  import initializers
# from keras.layers import advanced_activations
# from keras.models import Sequential
# from keras.utils.vis_utils import plot_model
# from keras import backend as K
import os.path

print('data_collection')
RAW_LOC = "../data/raw"
OUT_EXT = ".out"
FCH_EXT = ".fch"
OUT_LOC = os.path.join(RAW_LOC, "outs")
FCH_LOC = os.path.join(RAW_LOC, "fchks")
EXL_LOC = os.path.join(RAW_LOC, "X10_X17_WAVE2.xlsx")
# EXL_LOC = r".\data\raw\X10_X17_WAVE2.xlsx"
EXCLUDE_KEYS = [217, 216, 206, 205, 184, 183, 82, 81, 45]

def read_out(loc, num, ext=OUT_EXT):
    with open(os.path.join(loc, str(num)+ext), "r") as f:
        return f.read()

def read_fch(loc, num, ext=FCH_EXT):
    with open(os.path.join(loc, "Anth_"+str(num)+ext), "r") as f:
        return f.read()

def parse_x1(num, header=['Center Number', 'Atomic Number', 'Atomic Type', 'X', 'Y', 'Z'], raw_loc=OUT_LOC):
    # Extract the Standard Orientation
    # print(num,"check")
    outtxt = read_out(raw_loc, num)

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

    df[df.columns] = df[df.columns].apply(pd.to_numeric)

    return df


def parse_x2(num, header=['Alpha  occ. eigenvalues', 'Alpha  virt. eigenvalues'],
            raw_loc=OUT_LOC):
    # Extract the Alpha occ. & virt eigenvalues

    outtxt = read_out(raw_loc, num)

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


def parse_x3(num, header=["Atom Number", "Atom"], raw_loc=OUT_LOC):
    # Condensed to atoms (all electrons)
    outtxt = read_out(raw_loc, num)

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
            df = pd.read_csv(StringIO(str("\n".join(lines[data_line_start:idx]))), delim_whitespace=True, header=None, names=header + [str(i) for i in range(col_start,col_start+6)])

            # Remove duplicate Atom & Atom Number columns
            del df["Atom"]
            del df["Atom Number"]

            dataframes.append(df)

            # Adjust starting column
            col_start +=6

            # Adjust starting block
            data_line_start = idx + 1

    return pd.concat(dataframes, axis= 1).dropna(axis=1)


def parse_x4(num, header=["Val1", "Val2", "Val3"], raw_loc=OUT_LOC):
    # Electric Field Gradient, Eigenvalues
    outtxt = read_out(raw_loc, num)

    # Electric Field Gradient Eigenvalues
    header_expression = re.finditer(r"            Electrostatic Properties Using The SCF Density", outtxt)

    # Use the second occurrence in the file.
    # Some examples (like 228) only have a single occurrence, so use the final one.
    header_expression = list(header_expression)[-1]

    header_end = int(header_expression.end())

    # Find leading and closing barrier
    barrier_str = " **********************************************************************\n"
    leading_pos = outtxt.find(barrier_str, header_end)
    closing_pos = outtxt.find(" -----------------------------------------------------------------", leading_pos + len(barrier_str))
    a=StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1]))
    return pd.read_csv(a, delim_whitespace=True, header=None, names=header).reset_index()[header]

def parse_x5(num, header=["Electric Potential", "X", "Y", "Z"], raw_loc=OUT_LOC):
    # Electric Field Gradient, Eigenvalues
    outtxt = read_out(raw_loc, num)

    # Electric Field Gradient Eigenvalues
    header_expression = re.finditer(r"    Center     Electric         -------- Electric Field --------\n               Potential          X             Y             Z", outtxt)

    # Use the second occurrence in the file.
    # Some examples (like 228) only have a single occurrence, so use the final one.
    header_expression = list(header_expression)[-1]

    header_end = int(header_expression.end())

    # Find leading and closing barrier
    barrier_str = " -----------------------------------------------------------------"
    leading_pos = outtxt.find(barrier_str, header_end)
    closing_pos = outtxt.find(barrier_str, leading_pos + len(barrier_str))
    return pd.read_csv(StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1])), delim_whitespace=True, header=None, names=header).reset_index()[header]


def parse_x6(num, raw_loc=OUT_LOC):
    # Electric Field Gradient, Coordinates
    outtxt = read_out(raw_loc, num)

    # Electric Field Gradient Eigenvalues
    results = []
    for group in ["XX            YY            ZZ", "XY            XZ            YZ"]:
        header_expression = re.finditer(r"    Center         ---- Electric Field Gradient ----\n                     " + group, outtxt)

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
    df1 = pd.read_csv(StringIO(str(results[0])), delim_whitespace=True, header=None, names=["Atom Number", "XX", "YY", "ZZ"])
    df2 = pd.read_csv(StringIO(str(results[1])), delim_whitespace=True, header=None, names=["Atom Number", "XY", "XZ", "YZ"])
    del df2["Atom Number"]

    # Return vertically concatenated dataframes
    return pd.concat([df1, df2], axis= 1).iloc[:,1:]


def parse_x7(num, header=["Atom Number", "Eigen 1", "Eigen 2", "Eigen 3"], raw_loc=OUT_LOC):
    # Electric Field Gradient, Eigenvalues
    outtxt = read_out(raw_loc, num)

    # Electric Field Gradient Eigenvalues
    header_expression = re.search(r"    Center         ---- Electric Field Gradient ----\n                   ----       Eigenvalues       ----", outtxt)
    header_end = int(header_expression.end())

    # Find leading and closing barrier
    barrier_str = " -----------------------------------------------------"
    leading_pos = outtxt.find(barrier_str, header_end)
    closing_pos = outtxt.find(barrier_str, leading_pos + len(barrier_str))

    return pd.read_csv(StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1])), delim_whitespace=True, header=None, names=header).iloc[:,1:]


def parse_x8(num, raw_loc=FCH_LOC):
    # total SCF density
    outtxt = read_fch(raw_loc, num)


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


def parse_x9(num, raw_loc=FCH_LOC):
    outtxt = read_fch(raw_loc, num)

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


def parse_x10_through_x17(num, raw_loc=EXL_LOC):
    # Parses through the remaining data in the excel sheet.
    # Return the row for the data number
    # [u'Input', u'Key', u'Associated data', u'X10: Category Method', u'X11: Temperature (K)', u'X12: [Salt*Valency]', u'X13: Category Salt type', u'X14: [Buffer] (mM)', u'X15: pH', u'X16: CI #', u'X17: CI ', u'Unnamed: 11', u'Output: logK']

    dfs = pd.read_excel(raw_loc, sheet_name=None)['COMPUTER SCIENTISTS LOOK HERE']
    dfs = dfs.loc[dfs[u'Input'] == num]
    return dfs

def bucket(val,cutoffs):
        level = 0
        for k in cutoffs:
            # print(type(val)==str or val < k,"Check")
            if type(val)==str or val < k:
                # print(level)
                break
            level +=1
        # print(level)
        return level
def parse_master_file(raw_loc=EXL_LOC, exclude_keys=EXCLUDE_KEYS):
    ''' reads the master Excel file 'X10_X17_WAVE2.xlsx', drops rows where the
    `Key` column is within the list `exclude_keys`, and returns a DataFrame.'''


    df = pd.read_excel(raw_loc, sheet_name=None)['COMPUTER SCIENTISTS LOOK HERE']
    cutoffs = [3,9]
    print(cutoffs)
    df['Output: logKbucket']=df['Output: logK'].apply(lambda x: bucket(x,cutoffs))
    return df[~df['Key'].isin(exclude_keys)].reset_index(drop=True)
# def parse_x1(num, header=['Center Number', 'Atomic Number', 'Atomic Type', 'X', 'Y', 'Z'], raw_loc=OUT_LOC):
#     # Extract the Standard Orientation

#     outtxt = read_out(raw_loc, num)

#     # Find the final occurrence of "Standard orientation"
#     header_expression = re.finditer(r"Standard orientation", outtxt)
#     header_expression = list(header_expression)[-1]
#     header_end = int(header_expression.end())

#     # Find the starting boundary
#     block = " ---------------------------------------------------------------------"
#     boundary1 = outtxt.find(block, header_end)
#     boundary2 = outtxt.find(block, boundary1 + len(block))
#     boundary3 = outtxt.find(block, boundary2 + len(block))

#     # Extract all the data elements
#     values = re.split('\s{1,}|\n', outtxt[boundary2+len(block): boundary3])[1:-1]

#     # Group into blocks of 6
#     values = [values[a:a+6] for a in range(0,len(values),6)]

#     # Create final data frame & remove atomic type column.
#     df = pd.DataFrame(values, columns=header)
#     df = df.drop(columns=['Atomic Type'])

#     return df'.//.


# def parse_x2(num, header=['Alpha  occ. eigenvalues', 'Alpha  virt. eigenvalues'],
#             raw_loc=OUT_LOC):
#     # Extract the Alpha occ. & virt eigenvalues

#     outtxt = read_out(raw_loc, num)

#     # Find the final occurrence of "Alpha  occ. eigenvalues"
#     header_expression = re.finditer(r"( The electronic state is (.*).(\n)) (Alpha  occ. eigenvalues)", outtxt)
#     header_expression = list(header_expression)[-1]
#     header_end = int(header_expression.span(2)[1])

#     # Find the starting point for the "Alpha virt. eigenvalues"
#     virt_starting_pos = outtxt.find("Alpha virt. eigenvalues", header_end)
#     virt_ending_pos = outtxt.find("Condensed to atoms", virt_starting_pos)

#     # Prepare the blocks
#     occ_block = outtxt[header_end:virt_starting_pos]
#     virt_block = outtxt[virt_starting_pos:virt_ending_pos]

#     occ_block = map(lambda line: line[line.index("--")+2:] if "--" in line else "", occ_block.split("\r\n"))
#     virt_block = map(lambda line: line[line.index("--")+2:] if "--" in line else "", virt_block.split("\r\n"))

#     occ_block = " ".join(occ_block)
#     virt_block = " ".join(virt_block)

#     occ_block = re.split('\s{1,}', occ_block)[1:-1]
#     virt_block = re.split('\s{1,}', virt_block)[1:-1]

#     # Create final data frame & remove atomic type column.
#     df_occ = pd.DataFrame(occ_block, columns=["Alpha  occ. eigenvalues"])
#     df_virt = pd.DataFrame(virt_block, columns=["Alpha  virt. eigenvalues"])

#     return (df_occ, df_virt)


# def parse_x3(num, header=["Atom Number", "Atom"], raw_loc=OUT_LOC):
#     # Condensed to atoms (all electrons)
#     outtxt = read_out(raw_loc, num)

#     # Electric Field Gradient Eigenvalues
#     header_expression = re.finditer(r"          Condensed to atoms \(all electrons\):", outtxt)

#     # Use the second occurrence in the file.
#     # Some examples (like 228) only have a single occurrence, so use the final one.
#     header_expression = list(header_expression)[-1]

#     header_end = int(header_expression.end())

#     # Find leading and closing barrier
#     leading_pos = outtxt.find("\n", header_end)
#     closing_pos = outtxt.find(" Mulliken charges:", leading_pos)

#     # Split into separate lines
#     lines = outtxt[leading_pos+2:closing_pos].split("\n")[1:]

#     # Create each dataframe
#     dataframes = []
#     col_start = 1
#     data_line_start = 0
#     for idx, line in enumerate(lines):
#         if line.startswith("              ") or idx == len(lines)-1:
#             # We've reached an ending point
#             # df = pd.read_csv(StringIO(str("\n".join(lines[data_line_start:idx])).decode('utf-8')), delim_whitespace=True, header=None, names=header + [str(i) for i in range(col_start,col_start+6)])
#             df = pd.read_csv(StringIO(str("\n".join(lines[data_line_start:idx]))), delim_whitespace=True, header=None, names=header + [str(i) for i in range(col_start,col_start+6)])
#             # Remove duplicate Atom & Atom Number columns
#             del df["Atom"]
#             del df["Atom Number"]

#             dataframes.append(df)

#             # Adjust starting column
#             col_start +=6

#             # Adjust starting block
#             data_line_start = idx + 1

#     return pd.concat(dataframes, axis= 1).dropna(axis=1)


# def parse_x4(num, header=["Val1", "Val2", "Val3"], raw_loc=OUT_LOC):
#     # Electric Field Gradient, Eigenvalues
#     outtxt = read_out(raw_loc, num)
#     # print(outtxt)
#     # Electric Field Gradient Eigenvalues
#     header_expression = re.finditer(r"            Electrostatic Properties Using The SCF Density", outtxt)

#     # Use the second occurrence in the file.
#     # Some examples (like 228) only have a single occurrence, so use the final one.
#     header_expression = list(header_expression)[-1]

#     header_end = int(header_expression.end())

#     # Find leading and closing barrier
#     barrier_str = " **********************************************************************\n"
#     leading_pos = outtxt.find(barrier_str, header_end)
#     closing_pos = outtxt.find(" -----------------------------------------------------------------", leading_pos + len(barrier_str))
#     # a=StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1]).decode('utf-8'))
#     a=StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1]))
#     return pd.read_csv(a, delim_whitespace=True, header=None, names=header).reset_index()[header]


# def parse_x5(num, header=["Electric Potential", "X", "Y", "Z"], raw_loc=OUT_LOC):
#     # Electric Field Gradient, Eigenvalues
#     outtxt = read_out(raw_loc, num)
#     # print("")
#     # print("!!!!")
#     # print("")
#     # print("")
#     # print("")
#     # print("!!!!")
#     # print("")
#     # print("")
#     # print("")
#     # print("!!!!")
#     # print("")
#     # print("")
#     # print(repr(outtxt))
#     # Electric Field Gradient Eigenvalues
#     # header_expression = re.finditer(r"    Center     Electric         -------- Electric Field --------\r\n               Potential          X             Y             Z", outtxt)
#     header_expression = re.finditer(r"    Center     Electric         -------- Electric Field --------\n               Potential          X             Y             Z", outtxt)
#     # print(type(header_expression))
#     # Use the second occurrence in the file.
#     # Some examples (like 228) only have a single occurrence, so use the final one.
#     header_expression = list(header_expression)[-1]

#     header_end = int(header_expression.end())

#     # Find leading and closing barrier
#     barrier_str = " -----------------------------------------------------------------"
#     leading_pos = outtxt.find(barrier_str, header_end)
#     closing_pos = outtxt.find(barrier_str, leading_pos + len(barrier_str))
#     # removed .decode
#     return pd.read_csv(StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1])), delim_whitespace=True, header=None, names=header).reset_index()[header]
#     # return pd.read_csv(StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1]).decode('utf-8')), delim_whitespace=True, header=None, names=header).reset_index()[header]


# def parse_x6(num, raw_loc=OUT_LOC):
#     # Electric Field Gradient, Coordinates
#     outtxt = read_out(raw_loc, num)

#     # Electric Field Gradient Eigenvalues
#     results = []
#     for group in ["XX            YY            ZZ", "XY            XZ            YZ"]:
#         header_expression = re.finditer(r"    Center         ---- Electric Field Gradient ----\n                     " + group, outtxt)

#         # Use the second occurrence  in the file.
#         # Some examples (like 228) only have a single occurrence, so use the final one.
#         header_expression = list(header_expression)[-1]

#         # Get the ending position
#         header_end = int(header_expression.end())

#         # Find leading and closing barrier
#         barrier_str = " -----------------------------------------------------"
#         leading_pos = outtxt.find(barrier_str, header_end)
#         closing_pos = outtxt.find(barrier_str, leading_pos + len(barrier_str))

#         # Append to results
#         results.append(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1])

#     # Prepare Data Frames
#     ##PYTHON 3- removed utf
#     df1 = pd.read_csv(StringIO(str(results[0])), delim_whitespace=True, header=None, names=["Atom Number", "XX", "YY", "ZZ"])
#     df2 = pd.read_csv(StringIO(str(results[1])), delim_whitespace=True, header=None, names=["Atom Number", "XY", "XZ", "YZ"])
#     del df2["Atom Number"]

#     # df1 = pd.read_csv(StringIO(str(results[0]).decode('utf-8')), delim_whitespace=True, header=None, names=["Atom Number", "XX", "YY", "ZZ"])
#     # df2 = pd.read_csv(StringIO(str(results[1]).decode('utf-8')), delim_whitespace=True, header=None, names=["Atom Number", "XY", "XZ", "YZ"])
#     # del df2["Atom Number"]

#     # Return vertically concatenated dataframes
#     return pd.concat([df1, df2], axis= 1).iloc[:,1:]


# def parse_x7(num, header=["Atom Number", "Eigen 1", "Eigen 2", "Eigen 3"], raw_loc=OUT_LOC):
#     # Electric Field Gradient, Eigenvalues
#     outtxt = read_out(raw_loc, num)

#     # Electric Field Gradient Eigenvalues
#     header_expression = re.search(r"    Center         ---- Electric Field Gradient ----\n                   ----       Eigenvalues       ----", outtxt)
#     header_end = int(header_expression.end())

#     # Find leading and closing barrier
#     barrier_str = " -----------------------------------------------------"
#     leading_pos = outtxt.find(barrier_str, header_end)
#     closing_pos = outtxt.find(barrier_str, leading_pos + len(barrier_str))
#     return pd.read_csv(StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1])), delim_whitespace=True, header=None, names=header).iloc[:,1:]
#     # return pd.read_csv(StringIO(str(outtxt[leading_pos+len(barrier_str)+2:closing_pos-1]).decode('utf-8')), delim_whitespace=True, header=None, names=header).iloc[:,1:]


# def parse_x8(num, raw_loc=FCH_LOC):
#     # total SCF density
#     outtxt = read_fch(raw_loc, num)


#     # Using a regular expression, find the number of data elements.
#     header_expression = re.search(r"Total SCF Density                          R   N=(\s)+(\d+)", outtxt)
#     num_elems = int(header_expression.group(2))
#     header_end = int(header_expression.end())

#     # Extract all the data elements
#     values = re.split('\s{1,}|\n', outtxt[header_end:])[1:num_elems+1]

#     # Verify that each value is numeric.
#     for val in values:
#         try:
#             float(val)
#         except ValueError:
#             raise ValueError('A non-numeric value has been processed.')

#     return pd.DataFrame(values)


# def parse_x9(num, raw_loc=FCH_LOC):
#     outtxt = read_fch(raw_loc, num)

#     # Using a regular expression, find the number of data elements.
#     header_expression = re.search(r"Alpha MO coefficients                      R   N=(\s)+(\d+)", outtxt)
#     num_elems = int(header_expression.group(2))
#     header_end = int(header_expression.end())

#     # Extract all the data elements
#     values = re.split('\s{1,}|\n', outtxt[header_end:])[1:num_elems+1]

#     # Verify that each value is numeric.
#     for val in values:
#         try:
#             float(val)
#         except ValueError:
#             raise ValueError('A non-numeric value has been processed.')

#     return pd.DataFrame(values)


# def parse_x10_through_x17(num, raw_loc=EXL_LOC):
#     # Parses through the remaining data in the excel sheet.
#     # Return the row for the data number
#     # [u'Input', u'Key', u'Associated data', u'X10: Category Method', u'X11: Temperature (K)', u'X12: [Salt*Valency]', u'X13: Category Salt type', u'X14: [Buffer] (mM)', u'X15: pH', u'X16: CI #', u'X17: CI ', u'Unnamed: 11', u'Output: logK']

#     dfs = pd.read_excel(raw_loc, sheet_name=None)['COMPUTER SCIENTISTS LOOK HERE']
#     dfs = dfs.loc[dfs[u'Input'] == num]
#     return dfs

# def parse_master_file(raw_loc=EXL_LOC, exclude_keys=EXCLUDE_KEYS):
#     ''' reads the master Excel file 'X10_X17_WAVE2.xlsx', drops rows where the
#     `Key` column is within the list `exclude_keys`, and returns a DataFrame.'''
#     df = pd.read_excel(raw_loc, sheet_name=None)['COMPUTER SCIENTISTS LOOK HERE']
#     return df[~df['Key'].isin(exclude_keys)]

def get_filename_list(s):
    ''' takes a list-like object `s` and returns the numbers after SB_.'''

    return list(map(lambda x : str(x)[3:], s))

def get_dim_stats(l, f):
    if len(l) < 1:
        raise ValueError('l must be non-empty')

    spot = f(l[0])
    dim_counter = np.zeros(shape=(len(l), len(spot.shape)))
    xs = []

    for i in range(len(l)):
        ##work around for 901,902
        if l[i]!= '':
            x = f(l[i])
            dim_counter[i] = x.shape
            xs.append(x)

    return dim_counter, xs

def create_x1_matrix(l, max_dims=None, return_dims=False):
    dim_counter, xs = get_dim_stats(l, parse_x1)
    if max_dims is None:
        # print dim_counter
        mx = np.amax(dim_counter, axis=0)
        max_rows = int(mx[0])
        max_cols = int(mx[1])
    else:
        max_rows = max_dims[0]
        max_cols = max_dims[1]
    res = np.zeros(shape=(len(l), max_rows*max_cols))
    for i in range(len(xs)):
        for j in range(max_cols):
            res[i,max_rows*j:max_rows*j+xs[i].shape[0]] = xs[i].iloc[:,j].T
    if return_dims:
        return res, (max_rows, max_cols)
    return res

def create_x4_matrix(l, max_dims=None, return_dims=False):

    dim_counter, xs = get_dim_stats(l, parse_x4)
    if max_dims is None:
        # print dim_counter
        mx = np.amax(dim_counter, axis=0)
        max_rows = int(mx[0])
        max_cols = int(mx[1])
    else:
        max_rows = max_dims[0]
        max_cols = max_dims[1]
    res = np.zeros(shape=(len(l), max_rows*max_cols))
    for i in range(len(xs)):
        for j in range(max_cols):
            res[i,max_rows*j:max_rows*j+xs[i].shape[0]] = xs[i].iloc[:,j].T
    if return_dims:
        return res, (max_rows, max_cols)
    return res

def create_x5_matrix(l, max_dims=None, return_dims=False):

    dim_counter, xs = get_dim_stats(l, parse_x5)
    if max_dims is None:
        # print dim_counter
        mx = np.amax(dim_counter, axis=0)
        max_rows = int(mx[0])
        max_cols = int(mx[1])
    else:
        max_rows = max_dims[0]
        max_cols = max_dims[1]
    res = np.zeros(shape=(len(l), max_rows*max_cols))
    for i in range(len(xs)):
        for j in range(max_cols):
            res[i,max_rows*j:max_rows*j+xs[i].shape[0]] = xs[i].iloc[:,j].T
    if return_dims:
        return res, (max_rows, max_cols)
    return res


def create_x6_matrix(l, max_dims=None, return_dims=False):

    dim_counter, xs = get_dim_stats(l, parse_x6)
    if max_dims is None:
        # print dim_counter
        mx = np.amax(dim_counter, axis=0)
        max_rows = int(mx[0])
        max_cols = int(mx[1])
    else:
        max_rows = max_dims[0]
        max_cols = max_dims[1]
    res = np.zeros(shape=(len(l), max_rows*max_cols))
    for i in range(len(xs)):
        for j in range(max_cols):
            res[i,max_rows*j:max_rows*j+xs[i].shape[0]] = xs[i].replace(to_replace="************", value=0.0).iloc[:,j].T
    if return_dims:
        return res, (max_rows, max_cols)
    return res


def create_x7_matrix(l, max_dims=None, return_dims=False):

    dim_counter, xs = get_dim_stats(l, parse_x7)
    if max_dims is None:
        # print dim_counter
        mx = np.amax(dim_counter, axis=0)
        max_rows = int(mx[0])
        max_cols = int(mx[1])
    else:
        max_rows = max_dims[0]
        max_cols = max_dims[1]
    res = np.zeros(shape=(len(l), max_rows*max_cols))
    for i in range(len(xs)):
        for j in range(max_cols):
            res[i,max_rows*j:max_rows*j+xs[i].shape[0]] = xs[i].replace(to_replace="************", value=0.0).iloc[:,j].T
    if return_dims:
        return res, (max_rows, max_cols)
    return res

def create_data_item(num, exclude=[358,381], out_loc=OUT_LOC, excel_loc=EXL_LOC,
                    fch_loc=FCH_LOC):
    # Uses helper methods to create the data for item of given number.
    dfs_10_17 = parse_x10_through_x17(num)

    data_elements = {
        "x1" :  parse_x1(num, raw_loc=out_loc),
        "x2_occ" :  parse_x2(num, raw_loc=out_loc)[0],
        "x2_virt" :  parse_x2(num, raw_loc=out_loc)[1],
        "x3" :  parse_x3(num, raw_loc=out_loc) if num not in exclude else 0,
        "x4" :  parse_x4(num, raw_loc=out_loc),
        "x5" :  parse_x5(num, raw_loc=out_loc),
        "x6" :  parse_x6(num, raw_loc=out_loc),
        "x7" :  parse_x7(num, raw_loc=out_loc),
        "x8" :  parse_x8(num, raw_loc=fch_loc) if num in exclude else 0,
        "x9" :  parse_x9(num, raw_loc=fch_loc),
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

    return data_elements


def prepare_master(master):

    x10 = pd.get_dummies(master.iloc[:,0])
    if 'D, B, A' in x10.columns:
        mask = x10['D, B, A'] == 1
        x10.loc[mask,:] = [1, 1, 0, 1, 1]
        x10 = x10.drop(['D, B, A'], axis=1, inplace=False)

    x11 = master.iloc[:,1]
    x11 = x11.fillna(value=298)

    x12 = master.iloc[:,2]
    x12 = x12.fillna(value=0)

    x13 = master.iloc[:,3]
    x13 = x13.fillna(value=0)
    x13 = pd.get_dummies(x13)

    x14 = master.iloc[:,4]
    x14 = x14.fillna(value=0)

    x15 = master.iloc[:,5]
    x15 = x15.fillna(value=7.0)

    x16 = master.iloc[:,6]
    x16 = x16.fillna(value=0)
    x16 = pd.get_dummies(x16)

    x17 = master.iloc[:,7]
    x17 = x17.fillna(value='N')
    x17 = x17.replace(to_replace=' ', value='N')
    x17 = pd.get_dummies(x17)

    names = [x10.columns.values, [x11.name], [x12.name], x13.columns.values, [x14.name], [x15.name], x16.columns.values, x17.columns.values]
    flat_names = [val for sublist in names for val in sublist]

    return np.column_stack((x10.values, x11.values, x12.values, x13.values, x14.values, x15.values, x16.values, x17.values)), flat_names


def linear_regression_approx(x, y):
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    x_approx = regr.predict(x)
    return x_approx, regr
def linear_classification_approx(x,y):
    ##using sklearn SVM, can tune penalty function etc
    regr = SVC(kernel="linear")
    regr.fit(x, y)
    x_approx = regr.predict(x)
    return x_approx, regr
def regression_tree_approx(x,y, max_depth=2):
    regr = DecisionTreeRegressor(max_depth=max_depth)
    regr.fit(x, y)
    x_approx = regr.predict(x)
    return x_approx, regr
def decision_tree_approx(x,y, max_depth=2):
    regr = DecisionTreeClassifier(max_depth=max_depth)
    # regr = DecisionTreeClassifier(class_weight = {0: 2, 1: 1,2:1},max_depth=max_depth)
    regr.fit(x, y)
    x_approx = regr.predict(x)
    return x_approx, regr




def duplicate_master(master, threshold, times, how='lt', col_name='Output: logK'):
    if how == 'lt':
        to_dup = master[master[col_name] < threshold]
    elif how == 'gt':
        to_dup = master[master[col_name] > threshold]
    else:
        raise ValueError("please choose correct threshold comparator ('gt' or 'lt')")


    for i in range(times):
        master = pd.concat([master, to_dup], axis=0)

    return master


# def create_x1_matrix(l):

#     dim_counter, xs = get_dim_stats(l, parse_x1)
#     # print dim_counter
#     mx = np.amax(dim_counter, axis=0)
#     max_rows = int(mx[0])
#     max_cols = int(mx[1])
#     res = np.zeros(shape=(len(l), max_rows*max_cols))
#     for i in range(len(xs)):
#         for j in range(max_cols):
#             res[i,max_rows*j:max_rows*j+xs[i].shape[0]] = xs[i].iloc[:,j].T

#     return res

# def create_x4_matrix(l):

#     dim_counter, xs = get_dim_stats(l, parse_x4)
#     # print (dim_counter)

#     mx = np.amax(dim_counter, axis=0)
#     max_rows = int(mx[0])
#     max_cols = int(mx[1])
#     res = np.zeros(shape=(len(l), max_rows*max_cols))
#     # print(xs)
#     for i in range(len(xs)):
#         for j in range(max_cols):
#             # print("BEGIN",xs[i].iloc[:,j].T,"CHECKER")
#             # print("XSSTART",xs[i].iloc[:,0],"XS")
#             # print(i,j,"IJ/")
#             # print(xs[0].iloc[:,0].T)
#             # print(res[0,max_rows*0:max_rows*0+xs[i].shape[0]])

#             print(res[i,max_rows*j:max_rows*j+xs[i].shape[0]],"RESOLVED")
#             print(xs[i].iloc[:,j])
#             res[i,max_rows*j:max_rows*j+xs[i].shape[0]] = xs[i].iloc[:,j].T
#     # print(res)
#     return res

# def create_x5_matrix(l):

#     dim_counter, xs = get_dim_stats(l, parse_x5)
#     # print dim_counter
#     mx = np.amax(dim_counter, axis=0)
#     max_rows = int(mx[0])
#     max_cols = int(mx[1])
#     res = np.zeros(shape=(len(l), max_rows*max_cols))
#     # print(xs)
#     for i in range(len(xs)):
#         for j in range(max_cols):
#             res[i,max_rows*j:max_rows*j+xs[i].shape[0]] = xs[i].iloc[:,j].T
#     # print(res)
#     return res


# def create_x6_matrix(l):

#     dim_counter, xs = get_dim_stats(l, parse_x6)
#     # print dim_counter
#     mx = np.amax(dim_counter, axis=0)
#     max_rows = int(mx[0])
#     max_cols = int(mx[1])
#     res = np.zeros(shape=(len(l), max_rows*max_cols))
#     for i in range(len(xs)):
#         for j in range(max_cols):
#             res[i,max_rows*j:max_rows*j+xs[i].shape[0]] = xs[i].replace(to_replace="************", value=0.0).iloc[:,j].T

#     return res


# def create_x7_matrix(l):

#     dim_counter, xs = get_dim_stats(l, parse_x7)
#     # print(xs)
#     # print dim_counter
#     mx = np.amax(dim_counter, axis=0)
#     max_rows = int(mx[0])
#     max_cols = int(mx[1])
#     res = np.zeros(shape=(len(l), max_rows*max_cols))
#     for i in range(len(xs)):
#         for j in range(max_cols):
#             res[i,max_rows*j:max_rows*j+xs[i].shape[0]] = xs[i].replace(to_replace="************", value=0.0).iloc[:,j].T

#     return res

# def create_data_item(num, exclude=[358,381], out_loc=OUT_LOC, excel_loc=EXL_LOC,
#                     fch_loc=FCH_LOC):
#     # Uses helper methods to create the data for item of given number.
#     dfs_10_17 = parse_x10_through_x17(num)

#     data_elements = {
#         "x1" :  parse_x1(num, raw_loc=out_loc),
#         "x2_occ" :  parse_x2(num, raw_loc=out_loc)[0],
#         "x2_virt" :  parse_x2(num, raw_loc=out_loc)[1],
#         "x3" :  parse_x3(num, raw_loc=out_loc) if num not in exclude else 0,
#         "x4" :  parse_x4(num, raw_loc=out_loc),
#         "x5" :  parse_x5(num, raw_loc=out_loc),
#         "x6" :  parse_x6(num, raw_loc=out_loc),
#         "x7" :  parse_x7(num, raw_loc=out_loc),
#         "x8" :  parse_x8(num, raw_loc=fch_loc) if num in exclude else 0,
#         "x9" :  parse_x9(num, raw_loc=fch_loc),
#         "x10": dfs_10_17['X10: Category Method'],
#         "x11": dfs_10_17['X11: Temperature (K)'],
#         "x12": dfs_10_17['X12: [Salt*Valency]'],
#         "x13": dfs_10_17['X13: Category Salt type'],
#         "x14": dfs_10_17['X14: [Buffer] (mM)'],
#         "x15": dfs_10_17['X15: pH'],
#         "x16": dfs_10_17['X16: CI #'],
#         "x17": dfs_10_17['X17: CI '],
#         "output": dfs_10_17['Output: logK']
#     }

#     return data_elements


# def prepare_master(master):

#     x10 = pd.get_dummies(master.iloc[:,0])
#     if 'D, B, A' in x10.columns:
#         mask = x10['D, B, A'] == 1
#         x10.loc[mask,:] = [1, 1, 0, 1, 1]
#         x10 = x10.drop(['D, B, A'], axis=1, inplace=False)

#     x11 = master.iloc[:,1]
#     x11 = x11.fillna(value=298)

#     x12 = master.iloc[:,2]
#     x12 = x12.fillna(value=0)

#     x13 = master.iloc[:,3]
#     x13 = x13.fillna(value=0)
#     x13 = pd.get_dummies(x13)

#     x14 = master.iloc[:,4]
#     x14 = x14.fillna(value=0)

#     x15 = master.iloc[:,5]
#     x15 = x15.fillna(value=7.0)

#     x16 = master.iloc[:,6]
#     x16 = x16.fillna(value=0)
#     x16 = pd.get_dummies(x16)

#     x17 = master.iloc[:,7]
#     x17 = x17.fillna(value='N')
#     x17 = x17.replace(to_replace=' ', value='N')
#     x17 = pd.get_dummies(x17)

#     names = [x10.columns.values, [x11.name], [x12.name], x13.columns.values, [x14.name], [x15.name], x16.columns.values, x17.columns.values]
#     flat_names = [val for sublist in names for val in sublist]

#     return np.column_stack((x10.values, x11.values, x12.values, x13.values, x14.values, x15.values, x16.values, x17.values)), flat_names



# def linear_regression_approx(x, y):
#     # print(x,y,"end")
#     regr = linear_model.LinearRegression()
#     # print(y)
#     regr.fit(x, y)
#     print(regr.score(x,y),"end")
#     x1_approx = regr.predict(x)
#     # print(x1_approx)
#     return x1_approx

# def regression_tree_approx(x,y, max_depth=2):
#     regr = DecisionTreeRegressor(max_depth=max_depth)
#     regr.fit(x, y)
#     x_approx = regr.predict(x)
#     return x_approx, regr

# def NN_approx(x,y):
#     print(x[0,:],y)
#     model=Sequential()
#     # model.add(Dropout(.9))
#     # perfectly good
#     model.add(Dense(units=x.shape[1], activation='relu', input_dim=x.shape[1], kernel_initializer="random_normal"))
#     model.add(Dense(units=x.shape[1], activation='tanh',kernel_initializer="random_normal"))
#     # model.add(Dropout(.1))
#     model.add(Dense(units=x.shape[1], activation='relu',kernel_initializer="random_normal"))
#     # model.add(Dense(units=x.shape[1], activation='relu', input_dim=x.shape[1]))
#     # model.add(Dense(units=x.shape[1], activation='relu'))
#     # model.add(Dense(units=x.shape[1], activation='relu'))
#     # model.add(Dense(units=x.shape[1], activation='relu'))
#     # model.add(Dense(units=x.shape[1], activation='relu'))
#     # model.add(Dense(units=x.shape[1], activation='relu'))
#     # model.add(Dense(units=x.shape[1], activation='relu'))
#     # model.add(Dense(units=x.shape[1], activation='relu'))
#     # model.add(Dense(units=x.shape[1], activation='relu'))
#     # model.add(Dense(units=70, activation='relu',kernel_initializer="random_normal"))
#     # model.add(Dense(units=30, activation='tanh',kernel_initializer="random_normal"))
#     # model.add(Dense(units=20, activation='relu',kernel_initializer="random_normal"))
#     model.add(Dense(units=1, activation='linear'))
#     model.compile(optimizer='Adam', loss='mse')
#     model.fit(x,y,epochs=300,batch_size=x.shape[1]//3, verbose=1)
#     plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#     x_approx=model.predict_on_batch(x)
#     print(model.test_on_batch(x,y))
#     return x_approx
# def hypertest(x,y,acts,NUnits, Nlayers, Nepochs, testN,batch_size):
#     for t in testN:
#         model.add(Dense(units=x.shape[1], activation='relu', input_dim=x.shape[1]))
#         l=np.random.randint(0,len(Nlayers))
#         epochs=min(200,max(l*10,50))
#         for n in range(l):
#             a=np.random.randint(0,len(acts))
#             u=np.random.randint(0,len(uList))
#             model.add(Dense(units=u,activation=a))
#         model.fit(x,y,epochs,2)

# def duplicate_master(master, threshold, times, how='lt', col_name='Output: logK'):
#     if how == 'lt':
#         to_dup = master[master[col_name] < threshold]
#     elif how == 'gt':
#         to_dup = master[master[col_name] > threshold]
#     else:
#         raise ValueError("please choose correct threshold comparator ('gt' or 'lt')")

#     print ('to dup: ', to_dup.shape)
#     for i in range(times):
#         master = pd.concat([master, to_dup], axis=0)

#     return master



# def test_901902(trees,coeffs,ohone,ohtwo):

# for num in range(1,1000):
#     # Check if file exists
#     if os.path.isfile("data/raw/outs/" + str(num) + ".out") and os.path.isfile("data/raw/fchks/Anth_" + str(num) + ".fch"):
#         create_data_item(num)
