const container = d3.select("#grid");

// set the ranges
const color = d3.scaleSequential(d3.interpolateYlOrRd);

const tooltip = d3.select(".tooltip");
// const newsSource = d3.select(".tooltip.news-source");
const newsTitle = d3.select(".tooltip .news-title");
const newsSummary = d3.select(".tooltip .news-summary");

const now = new Date();
const path = `data/processed/${now.getFullYear()}/${now.getMonth()+1}/${now.getDate()}/de.json`;

d3.json(path).then(function(data) {
    color.domain(d3.extent(data, (d) => {return d.timestamp; }))

    container.selectAll(".item")
        .data(data)
        .enter()
        .append("div")
        .attr("class", "item")
        .style("grid-column", d => d.embedding[0]+1)
        .style("grid-row", d => d.embedding[1]+1)
        .style("border-color", d => color(d.timestamp))
        .html(d => `
            <h1 class="news-title">${d.title}</h1>
            <p class="news-summary">${d.summary}</p>
        `)
        // .on("mouseover", (event, d) => {
        //     tooltip.style("left", (event.pageX) + "px")
        //            .style("top", (event.pageY - 28) + "px");
        //     newsTitle.html(d.title);
        //     newsSummary.html(d.summary);
            
        //     tooltip.transition()
        //         .duration(200)
        //         .style("opacity", 1.0);
            
        // })
        // .on("mouseout", function(event, d) {
        //     tooltip.transition()
        //         .duration(500)
        //         .style("opacity", 0);
        // })
        .on("click", function(event, d) {
            const win = window.open(d.link, "_blank");
            win.focus();
        });
     
    // container.call(d3.zoom()
    //     .extent([[0, 0], [100, 100]])
    //     .scaleExtent([1, 8])
    //     // .translateExtent([[-width+100, -height+100], [2*width-100, 2*height-100]])
    //     .on("zoom", zoomed));
  
    // function zoomed({transform}) {
    //   container.style("transform", transform);
    // }
});