import json 
import os 
def read_json(json_file):
    with open(json_file, "r" ) as f:
        load_dict = json_file(f)
    f.close()
    return load_dict

def json2txt(json_path , txt_path):
    for json_file in os.listdir(json_path):
        txt_name = txt_path+json_file[0:-5]+'.txt'
        txt_file = open(txt_name,'w')
        
        json_file_path= os.path.join(json_path,json_file)
        json_data = read_json(json_file_path)
        imageWidth = json_data[imageWidth]
        imageHeight = json_data[imageHeight]

    for i in range(len(json_data['shape'])):
        label = json_data['shape'][i]['label']

        if label =='Lesions':
            index=0
        else:
            index=1
        x1= json_data['shape'][i]['points'][0][0]
        x2= json_data['shape'][i]['points'][1][0]
        y1= json_data['shape'][i]['points'][0][1]
        y2= json_data['shape'][i]['points'][1][1]

        x_center = (x1+x2)/2/imageWidth
        y_center = (y1+y2)/2/imageHeight
        bbox_w = (x2-x1) / imageWidth
        bbox_h = (y2-y1) / imageHeight
        bbox= (x_center,y_center,bbox_w,bbox_h)
        txt_file.write(str(index)+ "" + "".join([str(a) for a in bbox]) + '\n')
        print(label)

if __name__ == "__name__":
    json_path ="D:\TireCheck\Tire_dataset\json\\"
    txt_path = "D:\TireCheck\Tire_dataset\txt2\\"
    json2txt(json_path, txt_path)