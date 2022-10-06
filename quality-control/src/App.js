import Instructions from "./components/Instructions.js";
import React from "react";
import Submit from "./components/Submit.js";
import Task from "./components/Task.js";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

export default class App extends React.Component {
    constructor(props) {
        super(props);
        let data;
        if (!process.env.NODE_ENV || process.env.NODE_ENV === "development") {
            data =
                '[{"image": "https://upload.wikimedia.org/wikipedia/commons/1/15/White_Persian_Cat.jpg", "classIdx":"283", "background":"indoors"}, {"image": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Toiletpapier_%28Gobran111%29.jpg/1280px-Toiletpapier_%28Gobran111%29.jpg", "classIdx":"999"}]';
        } else {
            data = document.getElementById("inputData").value;
        }
        data = JSON.parse(data);
        this.images = data.map((x) => x.image);
        this.classIdxs = data.map((x) => x.classIdx);
        this.backgrounds = data.map((x) => x.background);
        let state = {
            idx: 0,
            nextDisabled: true,
            showInstructions: false,
            numChecked: 0,
            visited: new Set([0]),
        };
        for (let i = 0; i < this.images.length; i++) {
            state[this.images[i]] = this.createNewAttributes();
        }
        this.state = state;
    }

    createNewAttributes() {
        let newAttributes = {
            foreground: undefined,
            background: undefined,
            nsfw: undefined,
        };
        return newAttributes;
    }

    componentDidMount() {
        document.querySelector("crowd-form").onsubmit = () => {
            let annotations = {};
            for (let i = 0; i < this.images.length; i++) {
                annotations[this.images[i]] = this.state[this.images[i]];
            }
            document.getElementById("annotations").value =
                JSON.stringify(annotations);
        };
    }

    componentDidUpdate() {
        const attributes = this.state[this.images[this.state.idx]];
        for (let key in attributes) {
            if (attributes[key] === undefined) {
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
        const newVisited = new Set(this.state.visited);
        newVisited.add(newIdx);
        this.setState({
            idx: newIdx,
            visited: newVisited,
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
        let numChecked = this.state.numChecked;
        let newAttributes = this.createNewAttributes();
        const newOption = e.target.value;

        let prevAnnotated = 0;
        for (let attribute in this.state[image]) {
            if (this.state[image][attribute] !== undefined) {
                prevAnnotated++;
            }
        }

        for (let attribute in this.state[image]) {
            if (attribute === e.target.name) {
                newAttributes[attribute] = newOption;
            } else {
                newAttributes[attribute] = this.state[image][attribute];
            }
        }

        let nowAnnotated = 0;
        for (let attribute in this.state[image]) {
            if (this.state[image][attribute] !== undefined) {
                nowAnnotated++;
            }
        }

        numChecked += nowAnnotated - prevAnnotated;

        this.setState({
            [image]: newAttributes,
            numChecked: numChecked,
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
