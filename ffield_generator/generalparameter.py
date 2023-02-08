def generalparameter(type, value):
    """Version 1.2"""
    
    try:
        file = open("ffield","r+")
        file.seek(0)
        try:
            file.readline()
            
            for i in range(0,type):
                file.readline()
            position = file.tell()
            file.seek(position)
            file.write(value.rjust(10))
        
        finally:
            file.close()
    except IOError:
        pass
