import data from "./assets/input/inputData.json";
import Instructions from "./components/Instructions.js";
import React from "react";
import Submit from "./components/Submit.js";
import Task from "./components/Task.js";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

export default class App extends React.Component {
    constructor(props) {
        super(props);
        let inputData;
        if (!process.env.NODE_ENV || process.env.NODE_ENV === "development") {
            inputData = data;
        } else {
            inputData = document.getElementById("inputData").value;
            inputData = JSON.parse(inputData);
        }
        this.images = inputData.map((x) => x.image);
        if (!process.env.NODE_ENV || process.env.NODE_ENV === "development") {
            this.images = this.images.map((x) => "http://localhost:7777/" + x);
        }
        this.classIdxs = inputData.map((x) => x.classIdx);
        this.backgrounds = inputData.map((x) => x.background);
        this.hasInits = inputData.map((x) => x.hasInit);
        let state = {
            idx: 0,
            nextDisabled: true,
            showInstructions: false,
            visited: new Set(),
        };
        for (let i = 0; i < this.images.length; i++) {
            state[this.images[i]] = this.createNewAttributes(i);
        }
        this.state = state;
    }

    createNewAttributes(idx) {
        let newAttributes = {
            foreground: "",
            nsfw: "",
        };
        if (this.backgrounds[idx] !== undefined) {
            newAttributes["background"] = "";
        }
        if (this.hasInits[idx] !== undefined) {
            newAttributes["init"] = "";
        }
        return newAttributes;
    }

    componentDidMount() {
        if (process.env.NODE_ENV === "production") {
            document.querySelector("crowd-form").onsubmit = () => {
                let annotations = {};
                for (let i = 0; i < this.images.length; i++) {
                    annotations[this.images[i]] = this.state[this.images[i]];
                }
                document.getElementById("annotations").value =
                    JSON.stringify(annotations);
            };
        }
    }

    componentDidUpdate() {
        const attributes = this.state[this.images[this.state.idx]];
        for (let key in attributes) {
            if (attributes[key] === "") {
                if (!this.state.nextDisabled) {
                    this.setState({
                        nextDisabled: true,
                    });
                }
                return;
            }
        }
        if (
            this.state.nextDisabled &&
            this.state.idx !== this.images.length - 1
        ) {
            this.setState({
                nextDisabled: false,
            });
        }
    }

    handleInstructions = (e) => {
        if (this.state.showInstructions) {
            this.setState({
                showInstructions: false,
            });
        } else {
            this.setState({
                showInstructions: true,
            });
        }
    };

    updateIdxBy(change) {
        const newIdx = Math.min(
            Math.max(this.state.idx + change, 0),
            this.images.length - 1
        );
        this.setState({
            idx: newIdx,
        });
    }

    onPrev = () => {
        this.updateIdxBy(-1);
    };

    onNext = () => {
        if (!this.state.nextDisabled) {
            this.updateIdxBy(1);
        }
    };

    changeAttribute = (e) => {
        const image = this.images[this.state.idx];
        let visited = this.state.visited;
        let newAttributes = this.createNewAttributes(this.state.idx);
        const newOption = e.target.value;

        for (let attribute in this.state[image]) {
            if (attribute === e.target.name) {
                newAttributes[attribute] = newOption;
            } else {
                newAttributes[attribute] = this.state[image][attribute];
            }
        }

        let allAnnotated = true;
        for (let attribute in this.state[image]) {
            allAnnotated = allAnnotated && newAttributes[attribute] !== "";
        }

        if (allAnnotated) {
            visited.add(this.state.idx);
        }

        this.setState({
            [image]: newAttributes,
        });
    };

    render() {
        const progress = Math.ceil(
            (100 * this.state.visited.size) / this.images.length
        ).toString();
        const attributes = this.state[this.images[this.state.idx]];
        let componentToRender;
        let textOnInstructions;
        if (this.state.showInstructions) {
            componentToRender = <Instructions />;
            textOnInstructions = "Hide Instructions";
        } else {
            componentToRender = (
                <div>
                    <Task
                        classIdx={this.classIdxs[this.state.idx]}
                        imgSrc={this.images[this.state.idx]}
                        background={this.backgrounds[this.state.idx]}
                        hasInit={this.hasInits[this.state.idx]}
                        onChange={this.changeAttribute}
                        progress={progress}
                        prevDisabled={this.state.idx === 0}
                        nextDisabled={this.state.nextDisabled}
                        onPrev={this.onPrev}
                        onNext={this.onNext}
                        attributes={attributes}
                    />
                    {progress === "100" && <Submit />}
                </div>
            );
            textOnInstructions = "Show Instructions";
        }
        return (
            <div className="App">
                <nav
                    className="navbar navbar-expand-lg navbar-dark bg-dark"
                    id="header"
                >
                    <button
                        className="btn btn-outline-light my-2 my-sm-0"
                        onClick={this.handleInstructions}
                    >
                        {textOnInstructions}
                    </button>
                </nav>
                <div id="content">{componentToRender}</div>
            </div>
        );
    }
}
