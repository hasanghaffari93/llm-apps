Session: access to a Streamlit app in a browser tab
For each browser tab that connects to the Streamlit server, a new session is created.
Reloading a browser tab creates a new session.
Streamlit "reruns" your script from top to bottom every time you interact with your app.
Each reruns takes place in a blank slate: no variables are shared between runs.
Session State: a way to share variables between "reruns", for each user session. 
    
"st.cache_resource" is to cache “resources" that should be available globally across all users, sessions, and reruns.
Any mutations to the cached return value directly mutate the object in the cache

"st.cache_data"
Within each user session, an @st.cache_data decorated function returns a copy of the cached return value (if the value is already cached).

https://docs.streamlit.io/develop/concepts/architecture/caching

Each page is a script executed from your entrypoint file. \
You can define a page from a Python file or function. 
If you include elements or widgets in your entrypoint file, 
they become common elements between your pages.


Find good icon from https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded