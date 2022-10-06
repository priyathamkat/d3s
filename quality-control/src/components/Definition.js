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
                <h4>Definition</h4>
                {this.dictionary[this.props.clsIdx]}
                <hr />
            </div>
        );
    }
}
