import React from "react";
import "./Submit.css";

export default class Submit extends React.Component {
    downloadFile(data) {
        const blob = new Blob([data], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.download = "annotations.json";
        link.href = url;
        link.click();
        link.remove();
    }

    handleSubmit() {
        document.querySelector("crowd-form").submit();
        if (!process.env.NODE_ENV || process.env.NODE_ENV === 'development') {
            this.downloadFile(document.getElementById("annotations").value);
        }
    }

    render() {
        return (
            <div id="submit">
                <button
                    className="btn btn-primary btn-lg"
                    type="submit"
                    onClick={() => this.handleSubmit()}
                >
                    Submit
                </button>
            </div>
        );
    }
}
