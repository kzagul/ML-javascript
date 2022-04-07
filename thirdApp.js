document.getElementById("phrase").innerHTML = "HI"

const nn = ml5.neuralNetwork({
    task: "regression",
    debug: true,
});

nn.addData([1, 3], 4);
nn.addData([2, 4], 6);
nn.addData([1, 1], 1);
nn.addData([1, 7], 8);
nn.addData([4, 3], 7);
nn.addData([5, 5], 10);
nn.addData([5, 1], 6);
nn.addData([7, 7], 14);
nn.addData([6, 7], 13);
nn.addData([6, 2], 8);

nn.normalizeData();

nn.train({
    epochs: 50,
    batchSize: 10,
}, finishedTraining);

function finishedTraining() {
    document.getElementById("phrase").innerHTML = "Нейросеть обучилась";
    console.log(nn.classify([4, 3]));
}