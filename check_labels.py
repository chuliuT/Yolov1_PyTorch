import cv2

DATASET_PATH='./VOC2007/'
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']
           
def show_labels_img(imgname):
    """imgname是输入图像的名称，无下标"""
    img = cv2.imread(DATASET_PATH + "JPEGImages/" + imgname + ".jpg")
    h, w = img.shape[:2]
    print(w,h)
    label = []
    with open("./labels/"+imgname+".txt",'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            print(CLASSES[int(label[0])])
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
            cv2.putText(img,CLASSES[int(label[0])],pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            cv2.rectangle(img,pt1,pt2,(0,0,255,2))

    cv2.imshow("img",img)
    cv2.waitKey(0)
if __name__ == "__main__":
    show_labels_img('000125')