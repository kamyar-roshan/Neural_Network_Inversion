"""Read the params file and gets the ffield parameters as input"""



def params_input_train():
    try:
        file = open("params-ml","r")
        file_length = len(file.readlines())
        file.seek(0)
        parameters = []

        try:
            for i in range(file_length):
                parameters.append(file.readline().split())
        
        finally:
            file.close()
    except IOError:
        pass
        
    return parameters
