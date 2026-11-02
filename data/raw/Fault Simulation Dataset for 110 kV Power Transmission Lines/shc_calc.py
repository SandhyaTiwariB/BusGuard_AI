"""
Short-Circuit Calculation at Power Tranmission Lines in 110 kV Grid
Created on [29.04.2024]

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

lines = app.GetCalcRelevantObjects("*.ElmLne") # get objects of all(*) lines(.ElmLne) and store it as list

# get short circuit object 
sc = app.GetFromStudyCase("ComShc")
# short circuit settings
sc.iopt_mde = 1 # set method to IEC 60909
sc.iopt_shc = "3psc" # set fault type, 3psc is default.
fault_types = {
    "3psc": "3-Phase Short-Circuit",
    "2psc": "2-Phase Short-Circuit",
    "spgf": "Single Phase to Ground",
    "2pfg": "2-Phase to Ground"
}
sc.iopt_allbus = 0 # Fault Location set to User Selection
sc.shcobj = lines[0] # object to perform short circuit, we set it arbitrary to line1
sc.ppro = 50 # relative position of fault [%]

# creating object of the class ElmRes
# this type of object is used to store results after sc command execution
res = app.GetFromStudyCase("SC_results.ElmRes")
res.Clear() # purges data if any
# adding variables(fault position, line currents and voltages) and initialising the result file
res.AddVars(sc, "ppro")
for line in lines:
    res.AddVars(line, "m:Ikss:bus1:A", "m:Ikss:bus1:B", "m:Ikss:bus1:C") # initial sc current magnitudes at beginning of the line
    res.AddVars(line,  "n:U:bus1:A", "n:U:bus1:B", "n:U:bus1:C", "n:U:bus2:A", "n:U:bus2:B", "n:U:bus2:C") # line-to-ground voltage magnitudes at both side of the line
res.Init()

def execute_sc():
    '''
    executing short circuit faults of all types 
    through all lines in 110 kV grid
    at specified position with step of 5 %
    ---
    Returns: ElmRes obj
    ---
    '''
    for fault, fault_name in fault_types.items():
        sc.iopt_shc = fault
        for line in lines:
            sc.shcobj = line
            app.PrintPlain("---------------------------------------")
            app.PrintPlain(f"{fault_name} sweep at {line.loc_name}")
            app.PrintPlain("---------------------------------------")
            for i in range(21):
                sc.ppro = i * 5 # relative position of fault with step of 5 %
                sc.Execute() # execute short circuit
                res.Write()
    return res

# executing sc faults, writing to res file, closing
res = execute_sc()
res.Close()
# export from res to .csv file
app.ResLoadData(res)
export = app.GetFromStudyCase("SC_export.ComRes")
export.pResult = res
export.iopt_exp = 6
export.f_name = "sc_results.csv"
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
            
            
    





