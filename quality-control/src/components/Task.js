import data from "../assets/json/imagenet_classes.json";
import Metadata from "./Metadata.js";
import Option from "./Option.js";
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
            <div className="task">
                <div id="container">
                    <div id="task-pane">
                        <img
                            alt=""
                            id="task-img"
                            src="http://t2.gstatic.com/licensed-image?q=tbn:ANd9GcTC39YgxvHt_pubidqvYY9aMgaE9AGn2_ek8ah76x1IIwH1t0h9TS8A2w_s6A2ITbc-pjqzN56fKHj6Bjw"
                        />
                        <div id="questions">
                            <div className="question">
                                Is a {className} in the image?
                                <Option
                                    type="radio"
                                    value="1"
                                    name="foreground"
                                    id="radio-fg-yes"
                                    option="Yes"
                                />
                                <Option
                                    type="radio"
                                    value="0"
                                    name="foreground"
                                    id="radio-fg-no"
                                    option="No"
                                />
                            </div>
                            <div className="question">
                                Is the background in the image?
                                <Option
                                    type="radio"
                                    value="1"
                                    name="background"
                                    id="radio-bg-yes"
                                    option="Yes"
                                />
                                <Option
                                    type="radio"
                                    value="0"
                                    name="background"
                                    id="radio-bg-no"
                                    option="No"
                                />
                            </div>
                            <div className="question">
                                Is this an NSFW image?
                                <Option
                                    type="radio"
                                    value="1"
                                    name="nsfw"
                                    id="radio-nsfw-yes"
                                    option="Yes"
                                />
                                <Option
                                    type="radio"
                                    value="0"
                                    name="nsfw"
                                    id="radio-nsfw-no"
                                    option="No"
                                />
                            </div>
                        </div>
                    </div>
                    <Metadata
                        className={className}
                        classIdx={this.props.classIdx}
                    />
                </div>
                <div id="footer">
                    <div className="progress" id="progressBar">
                        <div
                            className="progress-bar"
                            role="progressbar"
                            style={{ width: this.props.progress + "%" }}
                        ></div>
                    </div>
                    <div className="btn-group nav-buttons" role="group">
                        <button
                            type="button"
                            className="btn btn-primary"
                            id="prev"
                            onClick={this.props.onPrev}
                            disabled={this.props.prevDisabled}
                        >
                            Previous
                        </button>
                        <button
                            type="button"
                            className="btn btn-primary"
                            id="next"
                            onClick={this.props.onNext}
                            disabled={this.props.nextDisabled}
                        >
                            Next
                        </button>
                    </div>
                </div>
            </div>
        );
    }
}
