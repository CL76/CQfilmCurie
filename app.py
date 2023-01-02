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
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
    col1, col2, col3 = st.columns( [0.4, 0.3,0.3])
    
    slider_crop_xmin = st.number_input("Origine x de la boite",0,pix.shape[1],0,step=1)
    slider_crop_xmax = st.number_input('taille x de la boite', 0, pix.shape[1], 500, step=1)
    slider_crop_ymin = st.number_input('Origine y de la boite', 0, pix.shape[0], 0, step=1)
    slider_crop_ymax = st.number_input('taille y de la boite', 0, pix.shape[0], 500, step=1)    
    
    
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
    with col3:
    

with tab2:
    st.image(FilmCQ_crop,width=150) 
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
    fig = plt.figure() 
    plt.plot([1, 2, 3, 4, 5]) 
    st.pyplot(fig)

with tab3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
    st.balloons()


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
