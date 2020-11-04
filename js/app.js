const svg = d3.select("#scatter");

const dimensions = svg.node().getBoundingClientRect();
const width = dimensions.width;
const height = dimensions.height;

const maxSize = Math.min(width, height);

const margins = {
    top: 10,
    right: 10,
    bottom: 10,
    left: 10
}

// set the ranges
const x = d3.scaleLinear().range([margins.left, maxSize - margins.right]);
const y = d3.scaleLinear().range([maxSize - margins.bottom, margins.top]);

var div = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

d3.json("data/processed/2020/11/4/de.json").then(function(data) {
    console.log(data);

    x.domain(d3.extent(data, (d) => {return d.low_dim_embedding[0]; }))
    y.domain(d3.extent(data, (d) => {return d.low_dim_embedding[1]; }))
    
    svg.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("r", 5)
        .attr("cx", d => x(d.low_dim_embedding[0]))
        .attr("cy", d => y(d.low_dim_embedding[1]))
        .on("mouseover", (event, d) => {
            console.log(d.title);
            div.html(d.title)
                .style("left", (event.pageX) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
});