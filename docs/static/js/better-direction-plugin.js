const betterDirectionPlugin = {
  id: 'betterDirectionArrow',
  afterDraw(chart) {
    const { ctx, chartArea, _ } = chart;
    ctx.save();
    ctx.strokeStyle = '#919191ff';
    ctx.fillStyle = '#919191ff';
    ctx.lineWidth = 3;
    ctx.font = '15px sans-serif';
    ctx.textAlign = 'center';

    const arrowX = chartArea.right + 20;
    const topY = chartArea.top;
    const bottomY = chartArea.bottom - 10;

    ctx.beginPath();
    ctx.moveTo(arrowX, topY);
    ctx.lineTo(arrowX, bottomY);
    ctx.stroke();

    const arrowHeadSize = 8;
    ctx.beginPath();
    ctx.moveTo(arrowX - arrowHeadSize / 2, bottomY);
    ctx.lineTo(arrowX + arrowHeadSize / 2, bottomY);
    ctx.lineTo(arrowX, bottomY + arrowHeadSize * 1.5);
    ctx.closePath();
    ctx.fill();

    ctx.translate(arrowX + 15, (chartArea.bottom + chartArea.top) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Better (Error is Lower)', 0, 0);

    ctx.restore();
  }
};
