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
const color = d3.scaleSequential(d3.interpolateYlOrRd);

const tooltip = d3.select("#tooltip");

const now = new Date();
const path = `data/processed/${now.getFullYear()}/${now.getMonth()+1}/${now.getDate()}/de.json`;

d3.json(path).then(function(data) {
    x.domain(d3.extent(data, (d) => {return d.low_dim_embedding[0]; }))
    y.domain(d3.extent(data, (d) => {return d.low_dim_embedding[1]; }))
    color.domain(d3.extent(data, (d) => {return d.timestamp; }))
    
    const g = svg.append("g")

    g.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("r", 5)
        .attr("cx", d => x(d.low_dim_embedding[0]))
        .attr("cy", d => y(d.low_dim_embedding[1]))
        .style("fill", d => color(d.timestamp))
        .on("mouseover", (event, d) => {
            tooltip.transition()
                .duration(200)
                .style("opacity", 1.0);
            tooltip.html(d.title)
                .style("left", (event.pageX) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(event, d) {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        })
        .on("click", function(event, d) {
            const win = window.open(d.link, "_blank");
            win.focus();
        });
     
    svg.call(d3.zoom()
        .extent([[0, 0], [width, height]])
        .scaleExtent([1, 8])
        .translateExtent([[0, 0], [width, height]])
        .on("zoom", zoomed));
  
    function zoomed({transform}) {
      g.attr("transform", transform);
    }
});