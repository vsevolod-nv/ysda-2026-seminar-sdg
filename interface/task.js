exports.Task = function (options) {
    TolokaHandlebarsTask.call(this, options);
    var boxes = this.getTask().input_values.initial_boxes;
    if (!Array.isArray(boxes)) {
        throw new Error("Вход initial_boxes должен содержать массив рамок и иметь hidden: false.");
    }
    var output = this.getSolution().output_values;
    if (!output.preannotation_initialized) {
        if (!Array.isArray(output.result) || output.result.length === 0) {
            this.setSolutionOutputValue("result", JSON.parse(JSON.stringify(boxes)));
        }
        this.setSolutionOutputValue("preannotation_initialized", true);
    }
};
exports.Task.prototype = Object.create(TolokaHandlebarsTask.prototype);
exports.Task.prototype.constructor = exports.Task;
exports.Task.prototype.onRender = function () {
    var task = this;
    this.getField("result").getEditor().on("shapes:update", function (shapes) {
        if (shapes.length === 0) {
            task.setSolutionOutputValue("result", []);
        }
    });
};
