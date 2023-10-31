import asyncio

from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.tools.code_interpreter import create_mock_client

from schema_agents.teams.image_analysis_hub.schemas import ReactUI, SoftwareRequirement

def create_web_developer(client=None):

    async def develop_react_ui(req: SoftwareRequirement, role: Role) -> ReactUI:
        """Develop the React UI plugin according to the software requirement, ensuring that it fulfills the desired functionality. """
        plugin = await role.aask(req, ReactUI)
        if client:
            await client.createWindow(
                src="https://gist.githubusercontent.com/oeway/b734c35f69a0ec0dcebe00b078676edb/raw/react-ui-plugin.imjoy.html",
                data={"jsx_script": plugin.jsx_script, "service_id": req.id}
            )
        return plugin

    web_developer = Role(name="Bob",
                                    profile="Web Developer",
                                    goal="Develop the React UI plugin according to the software requirement, ensuring that it fulfills the desired functionality. Implement necessary algorithms, handle data processing, and write tests to validate the correctness of the function.",
                                    constraints=None,
                                    actions=[develop_react_ui])
    return web_developer

def serve_plugin(plugin: ReactUI):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from jinja2 import Environment, PackageLoader, select_autoescape

    jinja_env = Environment(
        loader=PackageLoader("schema_agents"), autoescape=select_autoescape()
    )

    app = FastAPI(title="web app")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        temp = jinja_env.get_template("react_template.html")
        source = temp.render(plugin.dict())
        return source

    app.mount("/", index)
    uvicorn.run(app, host="127.0.0.1", port=8000)

async def main(context=None):
    mock_software_requirements = {
    "id": "cell_counter",
    "original_requirements": "Load an image file and segment the cells in the image, count the cells then show the result image. The cells are U2OS cells in a IF microscopy image, cells are round and in green color, the background is black.",
    "python_function_requirements": [
        {
        "function_name": "load_image",
        "function_signature": "load_image(image_path: str) -> np.ndarray",
        "feature_instructions": "This function should load an image from the provided path and return it as a numpy array. Use the OpenCV library for this.",
        "context_requirements": "The function will be used to load an image from a file path.",
        "testing_requirements": "Test the function with different image file paths to ensure it can handle different image formats and sizes."
        },
        {
        "function_name": "segment_cells",
        "function_signature": "segment_cells(image: np.ndarray) -> Tuple[np.ndarray, int]",
        "feature_instructions": "This function should segment the cells in the image and return the segmented image and the count of the cells. Use the OpenCV library for the segmentation. The cells are round and in green color, the background is black.",
        "context_requirements": "The function will be used to segment the cells in the image and count them.",
        "testing_requirements": "Test the function with different images to ensure it can accurately segment and count the cells."
        }
    ],
    "react_ui_requirements": {
        "plugin_name": "CellCounter",
        "ui_layout": "The UI should have an input field to enter the image file path, a button to load and segment the image, and a display area to show the segmented image and the cell count.",
        "interaction_patterns": "The user enters the image file path and clicks the button to load and segment the image. The segmented image and the cell count are then displayed.",
        "functionalities": "The UI should allow the user to enter the image file path, load and segment the image, and display the segmented image and the cell count.",
        "user_flows": "The user enters the image file path, clicks the button to load and segment the image, and sees the segmented image and the cell count."
    },
    "additional_notes": "The cells are U2OS cells in a IF microscopy image. The cells are round and in green color, the background is black. The number of cells in the image should be more than 12."
    }
    
    mock_plugin = {
        "plugin_name": "CellCounter",
        "root_element": "root",
        "react_version": "17.0.2",
        "react_dom_version": "17.0.2",
        "babel_version": "6.26.0",
        "jsx_script": """
            class CellCounter extends React.Component {
            constructor(props) {
                super(props);
                this.state = {
                imagePath: '',
                segmentedImage: null,
                cellCount: 0
                };
                this.handleImagePathChange = this.handleImagePathChange.bind(this);
                this.handleLoadAndSegment = this.handleLoadAndSegment.bind(this);
            }

            handleImagePathChange(event) {
                this.setState({imagePath: event.target.value});
            }

            async handleLoadAndSegment() {
                const image = await window.python.load_image(this.state.imagePath);
                const [segmentedImage, cellCount] = await window.python.segment_cells(image);
                this.setState({
                segmentedImage: segmentedImage,
                cellCount: cellCount
                });
            }

            render() {
                return (
                <div>
                    <input type='text' value={this.state.imagePath} onChange={this.handleImagePathChange} />
                    <button onClick={this.handleLoadAndSegment}>Load and Segment Image</button>
                    {this.state.segmentedImage && <img src={this.state.segmentedImage} />}
                    <p>Cell Count: {this.state.cellCount}</p>
                </div>
                );
            }
            }

            ReactDOM.render(
            <CellCounter />,
            document.getElementById('root')
            );
        """
    }
    # context['plugin'] = ReactUI.parse_obj(mock_plugin)
    WebDeveloper = create_web_developer(client=create_mock_client())
    wd = WebDeveloper()
    pr = SoftwareRequirement.parse_obj(mock_software_requirements)
    req = Message(content=pr.json(), data=pr, role="Project Manager")
    resp = await wd.handle(req)
    print(resp)
    if context is not None:
        context['plugin'] = resp[0].data
    

if __name__ == "__main__":
    context = {}
    asyncio.run(main(context))
    serve_plugin(context['plugin'])