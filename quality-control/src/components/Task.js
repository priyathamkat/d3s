import data from "../assets/json/imagenet_classes.json";
import Metadata from "./Metadata.js";
import React from "react";
import "./Task.css";

export default class Task extends React.Component {
    constructor(props) {
        super(props);
        this.classes = data;
    }
    render() {
        const className = this.classes[this.props.classIdx];
        return (
            <div id="container">
                <div id="task-pane">
                    <img
                        alt=""
                        id="task-img"
                        src="http://t2.gstatic.com/licensed-image?q=tbn:ANd9GcTC39YgxvHt_pubidqvYY9aMgaE9AGn2_ek8ah76x1IIwH1t0h9TS8A2w_s6A2ITbc-pjqzN56fKHj6Bjw"
                    />
                    <div id="questions">
                        <div>
                            Is a {className} in the image?
                            <input
                                type="radio"
                                name="foreground"
                                value="1"
                                id="radio-fg-yes"
                            ></input>
                            <label for="radio-fg-yes">Yes</label>
                            <input
                                type="radio"
                                name="foreground"
                                value="0"
                                id="radio-fg-no"
                            ></input>
                            <label for="radio-fg-no">No</label>
                        </div>
                        <div>
                            Is the background in the image?
                            <input
                                type="radio"
                                name="background"
                                value="1"
                                id="radio-bg-yes"
                            ></input>
                            <label for="radio-bg-yes">Yes</label>
                            <input
                                type="radio"
                                name="background"
                                value="0"
                                id="radio-bg-no"
                            ></input>
                            <label for="radio-bg-no">No</label>
                        </div>
                        <div>
                            Is this an NSFW image?
                            <input
                                type="radio"
                                name="nsfw"
                                value="1"
                                id="radio-nsfw-yes"
                            ></input>
                            <label for="radio-nsfw-yes">Yes</label>
                            <input
                                type="radio"
                                name="background"
                                value="0"
                                id="radio-nsfw-no"
                            ></input>
                            <label for="radio-nsfw-no">No</label>
                        </div>
                    </div>
                </div>
                <Metadata
                    className={className}
                    classIdx={this.props.classIdx}
                />
            </div>
        );
    }
}
