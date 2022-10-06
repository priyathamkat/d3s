import React from "react";
import "./Submit.css";

export default class Submit extends React.Component {
    handleSubmit() {
        document.querySelector("crowd-form").submit();
    }

    render() {
        return (
            <div id="submit">
                <button
                    className="btn btn-primary btn-lg"
                    type="submit"
                    onClick={this.handleSubmit}
                >
                    Submit
                </button>
            </div>
        );
    }
}
