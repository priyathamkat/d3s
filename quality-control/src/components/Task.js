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
        const clsName = this.classes[this.props.classIdx];
        return (
            <div className="task">
                <div id="container">
                    <div id="task-pane">
                        <img alt="" id="task-img" src={this.props.imgSrc} />
                        <div id="questions">
                            <div className="question">
                                Does the right image have the object that is in
                                the left image?
                                <Option
                                    type="radio"
                                    value="yes"
                                    name="foreground"
                                    id="radio-fg-yes"
                                    option="Yes"
                                    checked={
                                        this.props.attributes.foreground ===
                                        "Yes"
                                    }
                                    onChange={this.props.onChange}
                                />
                                <Option
                                    type="radio"
                                    value="no"
                                    name="foreground"
                                    id="radio-fg-no"
                                    option="No"
                                    checked={
                                        this.props.attributes.foreground ===
                                        "No"
                                    }
                                    onChange={this.props.onChange}
                                />
                            </div>
                            <div>
                                <em>
                                    (Some useful information about the object is
                                    in the panel on the right)
                                </em>
                            </div>
                            <div className="question">
                                Does the image have any adult content (nudity / extreme violence)?
                                <Option
                                    type="radio"
                                    value="yes"
                                    name="nsfw"
                                    id="radio-nsfw-yes"
                                    option="Yes"
                                    checked={
                                        this.props.attributes.nsfw === "Yes"
                                    }
                                    onChange={this.props.onChange}
                                />
                                <Option
                                    type="radio"
                                    value="no"
                                    name="nsfw"
                                    id="radio-nsfw-no"
                                    option="No"
                                    checked={
                                        this.props.attributes.nsfw === "No"
                                    }
                                    onChange={this.props.onChange}
                                />
                            </div>
                        </div>
                    </div>
                    <div id="metadata">
                        <Metadata
                            clsName={clsName}
                            classIdx={this.props.classIdx}
                        />
                    </div>
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
