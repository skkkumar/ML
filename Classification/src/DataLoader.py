'''
Created on 14-Jan-2018


'''
import csv
import re
from tabulate import tabulate


class DataLoader:
    """
    This class loads the data for machine learning samples
    """
    
    def getProperDataType(self, entry):
        """
        This method checks the datatype and converts to the proper
        python datatype
        Input entry is a string
        Assume three datatypes: float, int and string
        """
        if re.match("^\d+?\.\d+?$", entry) is not None:
            #float
            return float(entry), 1
        elif re.match("^\d+?$", entry) is not None:
            #integer
            return int(entry), 2
        else:
            #string
            return entry, 0
    
    
    def loadDataset(self, pathname, TYPE):
        """
        This method loads data from a specific pathname
        TYPE = 1, classification
        """
        
        #all data 
        data = []
        #unique values
        uniqueStrings = []
        #unique classes
        uniqueClasses = []
        
        #open file
        with open(pathname, 'rt') as csvfile:
            csvReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            
            #read the data line by line
            for line in csvReader:
                row = []
                
                #remove spaces from the line
                line = "".join(line)
                print ("line", line)
                
                if len(line) > 0:
                    #split the entries based on comma
                    elements = line.split(',')
                    for index in range(len(elements)):
                        #for every value in the line
                        entry = elements[index]
                        entry, dataType = self.getProperDataType(entry)
                        
                        #handle strings, convert to numeric data-type
                        if dataType == 0:
                            #if class
                            if TYPE == 1 and index == len(elements) - 1:
                                if entry in uniqueClasses:
                                    stringIndex = uniqueClasses.index(entry)
                                else:
                                    stringIndex = len(uniqueClasses)
                                    uniqueClasses.append(entry)
                            #if feature
                            else:
                                if entry in uniqueStrings:
                                    stringIndex = uniqueStrings.index(entry)
                                else:
                                    stringIndex = len(uniqueStrings)
                                    uniqueStrings.append(entry)
                            entry = stringIndex       
                                
                        row.append(entry)
                    data.append(row)

            if TYPE == 1:
                return data, uniqueClasses 
            

if __name__ == "__main__":
    loader = DataLoader()
    data, _ = loader.loadDataset("../data/fish.csv", 1)
    print (tabulate(data))
    
    
