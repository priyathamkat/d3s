import React from "react";
import Instructions from "./components/Instructions.js";
import Task from "./components/Task.js";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

export default class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            showInstructions: false,
        };
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
    render() {
        let componentToRender;
        let textOnInstructions;
        if (this.state.showInstructions) {
            componentToRender = <Instructions />;
            textOnInstructions = "Hide Instructions";
        } else {
            componentToRender = <Task classIdx="283" />;
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
