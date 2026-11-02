"""
Load Flow Calculation in 110 kV Grid
Created on [07.05.2024]

@author: Zivko Sokolovic
"""
# import the pf module and check if everything is fine.
import sys
try:
    PF_path = r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP2\Python\3.12"
    sys.path.append(PF_path)
    import powerfactory as pf
    app = pf.GetApplication()
    if app is None:
        raise Exception("Getting PowerFactory application failed !")
except Exception as error:
    print(error)
    exit(1)

app.ClearOutputWindow() # clears output window in PF 
app.EchoOff() 

def execute_ldf():
    '''
    executing load flow per each load active power change
    ---
    Returns: ElmRes obj
    ---
    '''
    for load in loads:
        initial_power = load.plini
        app.PrintPlain("---------------------------------------")
        app.PrintPlain(f"{load.loc_name}")
        app.PrintPlain("---------------------------------------")
        for i in range(101):
            load.plini = initial_power + i * 0.1 # active power increase by (i * 0.1 MW)
            ldf.Execute()
            res.Write()
        load.plini = initial_power

    return res

loads = app.GetCalcRelevantObjects("*.ElmLod") # get objects of all(*) loads(.ElmLod) and store it as list
lines = app.GetCalcRelevantObjects("*.ElmLne") # get objects of all(*) lines(.ElmLne) and store it as list 

# get load flow object 
ldf = app.GetFromStudyCase("ComLdf")
# load flow settings
ldf.iopt_net = 0 # AC Load Flow, balanced, positive sequence
ldf.iopt_pq = 0 # Consider Voltage Dependency of Loads, OFF

# creating object of the class ElmRes
# this type of object is used to store results after ldf command execution
res = app.GetFromStudyCase("LDF_results.ElmRes")
res.Clear() # purges data if any
# adding variables(Line currents and voltages) and initialising the result file
for line in lines:
    res.AddVars(line, "m:I:bus1:A", "m:I:bus1:B", "m:I:bus1:C") # current magnitudes at beginning of the line
    res.AddVars(line,  "n:U:bus1:A", "n:U:bus1:B", "n:U:bus1:C", "n:U:bus2:A", "n:U:bus2:B", "n:U:bus2:C") # line-to-ground voltage magnitudes at both side of the line
res.Init()

# executing load flow, writing to res file and closing
res = execute_ldf()
res.Close()
# export from res to .csv file
app.ResLoadData(res)
export = app.GetFromStudyCase("LDF_export.ComRes")
export.pResult = res
export.iopt_exp = 6
export.f_name = "ldf_results.csv"
export.numberFormat = 0
export.numberPrecisionFixed = 3
export.iopt_sep = False
export.col_Sep = ","
export.dec_Sep = "."
export.Execute()

# prints number of variables and values
NumVar = app.ResGetVariableCount(res)
NumVal = app.ResGetValueCount(res,0)
app.PrintPlain(res)
app.PrintPlain(f"Variables: {NumVar} Values: {NumVal}")

app.EchoOn()