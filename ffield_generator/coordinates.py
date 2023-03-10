

def coordinates():
    """Getting coordinates"""
    try:
        number_of_structures = 0
    
        with open('geo') as f:
            for line in f:
                finded = line.find('BIOGRF')
                if finded != -1:
                    number_of_structures += 1
    
        coordinates = []
    
    
        try:
            file = open("geo","r+")
            description = []
            position_end = 0
            
            for i in range(0,number_of_structures):
                position = file.read().find('DESCRP')
                file.seek(position+position_end)
                description.append(file.readline().split())
                coordinates.append(description[i][1])
                file.seek(position_end)
                
                position = file.read().find('FORMAT ATOM')
                file.seek(position + position_end)
                file.readline()
                position = file.tell()
                file.seek(position_end)
                position_conect = file.read().find('FORMAT CONECT') + position_end
                number_of_atoms = int((position_conect-position)/81)
                
                file.seek(position)
                
                for i in range(0,number_of_atoms):
                    coordinates.append(file.readline().split())
                
                
                position_end = file.read().find('END')+position_conect
                file.seek(position_end)
            
        finally:
            file.close()
    except IOError:
        pass
    
    return coordinates
