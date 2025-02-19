import streamlit as st
import time

st.set_page_config(
    page_title="Welcom to My Home",
    page_icon="📱"
)
st.title("Welcom to My Home")

st.markdown(
    """
# Hello!
Welcome to my FullstackGPT Portfolio!!

Here are the apps I made:

- [ ] [DocumentGPT](/DocumentGPT)
- [ ] [Art PsychologyGPT](/art_psychologyGPT)

"""
)


# with st.chat_message("human"):
#     st.write("Hello")

# with st.chat_message("ai"):
#     st.write("how ar you")

# with st.status("Embedding file...", expanded=True) as status:
#     time.sleep(2)
#     st.write("Getting the file")

#     time.sleep(2)
#     st.write("Embbeding the file")
    
#     time.sleep(2)
#     st.write("Caching the file")

#     status.update(label="Error", state="error")

# st.chat_input("Send a message to the ai")



# with st.sidebar:
#     st.title("Choose Contents")
#     st.text_input("xxx")

# tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])
# with tab_one:
#     st.write('a')

# with tab_two:
#     st.write('b')

# with tab_three:
#     st.write('c')