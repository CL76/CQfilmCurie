#Import libraries.  https://towardsdatascience.com/create-a-photo-converter-app-using-streamlit-surprisingly-easy-and-fun-db291b5010c6
#https://blog.streamlit.io/make-your-st-pyplot-interactive/
import streamlit as st
import cv2
import numpy as np
from  PIL import Image, ImageEnhance
import matplotlib.pyplot as plt


def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    
def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    Returns two arrays
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    """
    maxtab = []
    mintab = [] 
    if x is None:
        x = arange(len(v))    
    v = asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')   
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    lookformax = True
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    return array(maxtab), array(mintab)

    if __name__=="__main__":
        from matplotlib.pyplot import plot, scatter, show
        series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
        maxtab, mintab = peakdet(series,.3)
        plot(series)
        scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
        scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
        show()
    
    
def erode_NG_sketch_image(image,kernel_erode_x=10,kernel_erode_y=10,kernel_gaussianblur=101):
    kernel = np.ones((kernel_erode_x,kernel_erode_y), np.uint8) *255
    image_erode = cv2.erode(image, kernel)
    image_NG = cv2.cvtColor(image_erode, cv2.COLOR_RGB2GRAY)
    inv_gray = 255 - image_NG
    plt.imshow(inv_gray, )
    print(inv_gray.shape)
    blur_image = cv2.GaussianBlur(inv_gray, (kernel_gaussianblur,kernel_gaussianblur), 0, 0)
    plt.imshow(blur_image,)
    sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
    return sketch





def show_image_cadre(image, debut_y=100, fin=200, droite=200, gauche=100, title='', cmap_type='gray',alpha = 0.2):
    figure(figsize=(15, 15))
    plt.imshow(image, cmap=cmap_type)
    # Get the current reference
    ax = plt.gca()
    # Create a Rectangle patch
    rect = Rectangle((gauche,debut),image.shape[0]-droite-gauche,fin-debut,linewidth=1,edgecolor='r',alpha = alpha,facecolor='orchid')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.title(title)
    plt.axis('off')



    
    
def detect_peak(s_profil,value_peak_to_detect=1500,max_peak_to_remove=8000):
    #value_peak_to_detect = 1500
    maxtab, mintab = peakdet(s_profil,value_peak_to_detect)
    a_max = array(maxtab)[:,0]
    b_max = array(maxtab)[:,1]
    a_min = array(mintab)[:,0].tolist()
    b_min = array(mintab)[:,1].tolist()
    #plot(s)
    #scatter(a, b, color='blue')
    #show()
    #max_peak_to_remove=8000
    c_max = np.where( b_max < max_peak_to_remove)[0].tolist()
    aa_max=a_max.tolist()
    bb_max=b_max.tolist()
    d_max=[]
    e_max=[]
    for i in c_max:
    #print(i)
        d_max.append(aa_max[i])
        e_max.append(bb_max[i])
    plot(s_profil)
    scatter(d_max,e_max, color='lightskyblue')
    scatter(a_min,b_min, color='plum')
    show()
    liste_peak_max=[d_max,e_max]
    liste_peak_min = [a_min,b_min]
    return liste_peak_max,liste_peak_min



def rotate_image(image, angle):
    #from scipy import ndimage
    #image_center = tuple(np.array(image.shape[1::-1]) / 2)
    #rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    #result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    #return result    

    #rotation angle in degree
    imagerotate = ndimage.rotate(image, angle)
    show_image(imagerotate, 'Original RGB image');
    figure(figsize=(15, 15))
    plt.imshow(imagerotate, cmap='gray',)
    plt.axhline(y = imagerotate.shape[0]/2-100, color = 'orange', linestyle = '-')
    plt.axvline(x = 700, color = 'orange', linestyle = '-')
    plt.title("")
    plt.axis('off');    
    
    
    
def show_image_cadre_2(image, Origine_x_rect =400, Origine_y_rect =200,taille_x_rect = 500, taille_y_rect = 600, title='', cmap_type='gray',alpha = 0.2):
    figure(figsize=(15, 15))
    plt.imshow(image, cmap=cmap_type)
    # Get the current reference
    ax = plt.gca()
    # Create a Rectangle patch
    rect = Rectangle((Origine_x_rect,Origine_y_rect),taille_x_rect,taille_y_rect,linewidth=1,edgecolor='r',alpha = alpha,facecolor='orchid')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.title(title)
    plt.axis('off')    
    






#image = Image.open(r'...\Insights_Bees_logo.png') #Brand logo image (optional)

#Create two columns with different width
#col1, col2 = st.columns( [0.8, 0.2])
#with col1:               # To display the header text using css style
#    st.markdown(""" <style> .font {
#    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
#    </style> """, unsafe_allow_html=True)
#    st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)
    
#with col2:               # To display brand logo
#    st.image(image,  width=150)

#Add a header and expander in side bar
#st.sidebar.markdown('<p class="font"> Analyse de Films Permadoc - HDD - Baclesse</p>', unsafe_allow_html=True)
#with st.sidebar.expander("About the App"):
#     st.write("""
#        Charger la photo et blabla.  \n  \nCette app a été créée par Cédric LOISEAU - Décembre 2022
#     """)
#Add file uploader to allow users to upload photos
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])    

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    pix = np.array(image)
    #st.write(pix)
    st.write(pix.shape)
    
    


tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
    st.header("A cat")
    #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
    col1, col2, col3 = st.columns( [0.4, 0.3,0.3])
    
    slider_crop_xmin = st.number_input("Origine x de la boite",0,pix.shape[1],0,step=1)
    slider_crop_xmax = st.number_input('taille x de la boite', 0, pix.shape[1], pix.shape[1], step=1)
    slider_crop_ymin = st.number_input('Origine y de la boite', 0, pix.shape[0], 0, step=1)
    slider_crop_ymax = st.number_input('taille y de la boite', 0, pix.shape[0], pix.shape[0], step=1)    
    
    
    #slider_crop_xmin = st.slider('Origine x de la boite', 0, pix.shape[1]-1, 80, step=1)
        #slider_crop_xmin = st.sidebar.slider('x_min boite', 0, 500, 80, step=1)
    #slider_crop_xmax = st.slider('taille x de la boite', 0, pix.shape[1]-1, 200, step=1)
    #slider_crop_ymin = st.slider('Origine y de la boite', 0, pix.shape[0]-1, 80, step=1)
    #slider_crop_ymax = st.slider('taille y de la boite', 0, pix.shape[0]-1, 200, step=1)  
    with col1:
        st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
        st.image(pix,width=150)  
    with col2:
        FilmCQ_crop = pix[slider_crop_xmin:slider_crop_xmax, slider_crop_ymin:slider_crop_ymax]
        st.image(FilmCQ_crop,width=150)  
    #with col3:
    

with tab2:
    st.image(FilmCQ_crop,width=150) 
    st.header("image reduite")
    #st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
    fig = plt.figure() 
    plt.plot([1, 2, 3, 4, 5]) 
    st.pyplot(fig)

with tab3:
    st.header("i")
    #st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
    Origine_x_rect =400
    Origine_y_rect =200
    #taille_x_rect = 100
    taille_y_rect = 3400
    taille_x_rect = st.number_input('taille x de la boite', 0, imagerotate.shape[1], 100, step=1)
    show_image_cadre_2(imagerotate,Origine_x_rect, Origine_y_rect,taille_x_rect, taille_y_rect ,alpha = 0.4 );
    #st.balloons()
    

#Add 'before' and 'after' columns

    

        
        

        
        #create your figure and get the figure object returned

       # show_image(FilmCQ_crop, 'Original RGB image');

    #with col2:
    #    st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)
    #    filter = st.sidebar.radio('Covert your photo to:', ['Original','Erode Image','Gray Image','Black and White', 'Pencil Sketch', 'Blur Effect'])
    #    if filter == 'Gray Image':
    #            converted_img = np.array(image.convert('RGB'))
    #            gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
    #            st.image(gray_scale, width=150)
                
#        if filter == 'Erode Image':
#                converted_img = np.array(image.convert('RGB'))
                
                
# Creating kernel 
    
#                kernel = np.ones((10,10), np.uint8) *255
                
#                gray_scale = cv2.erode(converted_img, kernel,cv2.COLOR_RGB2GRAY)
#                st.image(gray_scale, width=300)        
       
#        elif filter == 'Black and White':
#                converted_img = np.array(image.convert('RGB'))
#                gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
#                slider = st.sidebar.slider('Adjust the intensity', 1, 255, 127, step=1)
#                (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, slider, 255, cv2.THRESH_BINARY)
#                st.image(blackAndWhiteImage, width=300)
#        elif filter == 'Pencil Sketch':
#                converted_img = np.array(image.convert('RGB')) 
#                gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
#                inv_gray = 255 - gray_scale
#                slider = st.sidebar.slider('Adjust the intensity', 25, 255, 125, step=2)
#                blur_image = cv2.GaussianBlur(inv_gray, (slider,slider), 0, 0)
#                sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
#                st.image(sketch, width=300) 
#        elif filter == 'Blur Effect':
#                converted_img = np.array(image.convert('RGB'))
#                slider = st.sidebar.slider('Adjust the intensity', 5, 81, 33, step=2)
#                converted_img = cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR)
#                blur_image = cv2.GaussianBlur(converted_img, (slider,slider), 0, 0)
#                st.image(blur_image, channels='BGR', width=300) 
#        else: 
#                st.image(image, width=300)
                
#Add a feedback section in the sidebar
#st.sidebar.title(' ') #Used to create some space between the filter widget and the comments section
#st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
#st.sidebar.subheader('Please help us improve!')
#with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
#    rating=st.slider("Please rate the app", min_value=1, max_value=5, value=3,help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
#    text=st.text_input(label='Please leave your feedback here')
#    submitted = st.form_submit_button('Submit')
#    if submitted:
#      st.write('Thanks for your feedback!')
#      st.markdown('Your Rating:')
#      st.markdown(rating)
#      st.markdown('Your Feedback:')
#      st.markdown(text)
