import cv2
import pyrealsense2 as rs
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

pipeline = rs.pipeline()
config = rs.config()
config.enable_device('243122301319')
config.enable_stream(rs.stream.color, 1280,720, rs.format.bgra8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1280,720, rs.format.y8,30)
pipeline.start(config)
colorizer=rs.colorizer()

    

def case_one():
   
    frames_wait = pipeline.wait_for_frames()

    color_frame = frames_wait.get_color_frame()
    coloring_image = np.asarray(color_frame.get_data())

    try:
        for i in iter(int,1):
          
            frames_wait = pipeline.wait_for_frames()
            color_frame = frames_wait.get_color_frame()
            
            Depth_frame= frames_wait.get_depth_frame()
            Depth_image=np.asanyarray(Depth_frame.get_data())
            

           
            coloring_image = np.asarray(color_frame.get_data())
            
            
            graying_img = cv2.cvtColor(coloring_image, cv2.COLOR_BGR2GRAY)

            faces_D = face_cascade.detectMultiScale(graying_img,1.05,25)
            
          
            for (a,b,c,d) in faces_D:
                cv2.rectangle(coloring_image, (a,b), (a+c, b+d), (0, 255, 0), 2)
                #cv2.circle(coloring_image,50,(a,b), (a+c, b+d), (0, 255, 0), 2)
                
                      
            depth_roi = np.mean(Depth_image[b:b+d, a:a+c])
            distance = np.mean(depth_roi)/1000
            #distance.delay(200)
           
            text_scr = f"Distance to apper: {distance:.2f} m"
            cv2.putText(coloring_image, text_scr, (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)

            filename=f"colour_image.jpg"
            cv2.imwrite(filename,coloring_image)       
            cv2.imshow('Face Detection for RealSense', coloring_image)

            if cv2.waitKey(1) == ord('a'):
                break
    finally:
        pipeline.stop()

      
def case_two():
    try:
        for i in iter(int,1):
            
            frames_wait = pipeline.wait_for_frames()
          
            Depth_frame= frames_wait.get_depth_frame()
            Depth_image=np.asanyarray(Depth_frame.get_data())
   
            Depth_colormap=cv2.applyColorMap((Depth_image/8000.0*255).astype(np.uint8),cv2.COLORMAP_RAINBOW)
            
            decimation = rs.decimation_filter()
            decimated_depth = decimation.process(Depth_frame)
           
            colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
            plt.imshow(colorized_depth)
            
            filename=f"Depth_image.jpg"
            cv2.imwrite(filename,Depth_image)


            cv2.imshow("Depth_image",Depth_colormap)
            if cv2.waitKey(1)==('a'):
                break
    finally:

        pipeline.stop()
    
    

def case_three():
   
 
    try:
        for i in iter(int,1):
          
            frames_wait = pipeline.wait_for_frames()
            colour_frame = frames_wait.get_color_frame()

           
            coloring_image = np.asarray(colour_frame.get_data())
            Depth_frame= frames_wait.get_depth_frame()
            Depth_image=np.asanyarray(Depth_frame.get_data())


            graying_img = cv2.cvtColor(coloring_image, cv2.COLOR_BGR2GRAY)

            faces_shape = face_cascade.detectMultiScale(graying_img,1.15,15)
            Depth_colormap=cv2.applyColorMap((Depth_image/8000.0*255).astype(np.uint8),cv2.COLORMAP_RAINBOW)

          
            for (a,b,c,d) in faces_shape:
                cv2.rectangle(coloring_image, (a,b), (a+c, b+d), (0, 255, 0), 2)

            depth_roi = Depth_image[b:b+d, a:a+c]
            distance = np.mean(depth_roi)/10
           
            text = f"Distance to apper: {distance:.2f} cm"
            cv2.putText(coloring_image, text, (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            filename=f"colour_image.jpg"
            cv2.imwrite(filename,coloring_image)       
            cv2.imshow('Face Detection', coloring_image)
            #coloring_image.wait(1)
            cv2.imshow("Depth_image",Depth_colormap)

            if cv2.waitKey(1) == ord('a'):
                break
    finally:
        pipeline.stop()
    
def case_four():
   
    try:
       for i in iter(int,1):

        
            frames_wait= pipeline.wait_for_frames()
        
            infrared_frame=frames_wait.get_infrared_frame()
            infrared_image=np.asanyarray(infrared_frame.get_data())
    
            infrared_colormap=cv2.applyColorMap(infrared_image,cv2.COLORMAP_RAINBOW)
    
            cv2.imshow("Infra", infrared_image)
            
            if cv2.waitKey(1)==ord('a'):
                break
            
    finally:
    
        pipeline.stop()
        
    
def case_default():
    print("Invalid selection \n\nLucky for you that you can still See the RGB Image....")
    case_one()
    
    
window=tk.Tk()


button1=tk.Button(window , text='RGB',command=case_one)
button1.pack()    
    
button2=tk.Button(window , text='Depth_Image',command=case_two)
button2.pack()   

button3=tk.Button(window , text='RGB+Depth',command=case_three)
button3.pack()   

button4=tk.Button(window , text='Infrared',command=case_four)
button4.pack() 

button5=tk.Button(window , text='CaseDefault',command=case_default)
button5.pack() 


window.mainloop()



cv2.destroyAllWindows()
