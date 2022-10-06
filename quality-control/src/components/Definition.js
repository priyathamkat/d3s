import data from "../assets/json/imagenet_dictionary.json";
import React from "react";
import "./Definition.css";

export default class Definition extends React.Component {
    constructor(props) {
        super(props);
        this.dictionary = data;
    }

    render() {
        return (
            <div>
                <em>{this.dictionary[this.props.clsIdx]}</em>
            </div>
        );
    }
}
