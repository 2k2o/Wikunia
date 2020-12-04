const svg = d3.select("#svg");
const g = d3.select("#main-group");

const dimensions = svg.node().getBoundingClientRect();
const width = dimensions.width;
const height = dimensions.height;

const maxSize = Math.min(width, height);

const margin = 30;

// set the ranges
const scale = d3.scaleLinear().range([margin, maxSize - margin]);
// const x = d3.scaleLinear().range([margins.left, maxSize - margins.right]);
// const y = d3.scaleLinear().range([maxSize - margins.bottom, margins.top]);

const opacity = d3.scaleLinear([0.1, 1]);

const tooltip = d3.select("#tooltip");
// const newsSource = d3.select(".tooltip.news-source");
const newsTitle = d3.select("#tooltip #news-title");
const newsSummary = d3.select("#tooltip #news-summary");

const now = new Date();
const url = "https://opensourc.es/Wikunia/data/processed/de.json";

function getHexagonString(d) {
    const dX = d.embedding[0] + 0.5;
    const dY = d.embedding[1] + 0.433;

    // we begin at the center of the hexagon
    // sin(60deg)*0.5
    offsetsX = [-0.5, -0.25, 0.25, 0.5, 0.25, -0.25];
    offsetsY = [0, -0.433, -0.433, 0, 0.433, 0.433];

    let points = "";
    for(let i=0; i < offsetsX.length; i++) {
        points += scale(dX + offsetsX[i]) + "," + scale(dY + offsetsY[i]) + " ";
    }

    return points;
}

d3.json(url).then(function(data) {
    opacity.domain(d3.extent(data, (d) => {return d.timestamp; }));
    // x.domain(d3.extent(data, (d) => {return d.embedding[0]; }));
    // y.domain(d3.extent(data, (d) => {return d.embedding[1]; }));
    scale.domain(d3.extent(data, (d) => {return d.embedding[0] + 0.5; }));

    g.selectAll("polygon")
        .data(data)
        .enter()
        .append("polygon")
        .attr("points", getHexagonString)
        .style("opacity", d => opacity(d.timestamp))
        .on("mouseover", (event, d) => {
            tooltip.style("left", (event.pageX + 10) + "px")
                   .style("top", (event.pageY + 10) + "px");
            newsTitle.html(d.title);
            newsSummary.html(d.summary);
            
            tooltip.style("opacity", 1.0);            
        })
        .on("mouseout", function(event, d) {
            tooltip.style("opacity", 0);
        })
        .on("click", function(event, d) {
            const win = window.open(d.link, "_blank");
            win.focus();
        });

    const g_dimensions = g.node().getBoundingClientRect();
    const g_width = g_dimensions.width;
    const g_height = g_dimensions.height;

    const zoom = d3.zoom()
        // .extent([[0, 0], [width, height]])
        .scaleExtent([0, 10])
        .translateExtent([[-width+100, -height+100], [2*width-100, 2*height-100]])
        .on("zoom", zoomed);

    // We set the initial translation in such a way that the hexagons are distributed around the center
    svg
        .call(zoom.transform, d3.zoomIdentity.translate(width/2 - g_width/2 - margin, height/2 - g_height/2 - margin))
        .call(zoom);
  
    function zoomed({transform}) {
        g.style("transform", "translate(" + transform.x + "px," + transform.y + "px) scale(" + transform.k + ")");
    }
});
