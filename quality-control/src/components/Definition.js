import React from "react";
import data from "../assets/json/imagenet_dictionary.json"

export default class Definition extends React.Component {
    constructor(props) {
        super(props);
        this.dictionary = data;
    }
    render() {
        return (
            <div>
                <i>a shuttlecock</i>
                <p>{this.dictionary[this.props.clsName]}</p>
            </div>
        );
    }
}
