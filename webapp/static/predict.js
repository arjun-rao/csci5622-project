function show_predictions(text, wts) {
  var color = "255, 70, 50";
  var useColor = $('#use_color').is(':checked');
  var useLines = $('#use_lines').is(':checked');
  var tokens = text.split(" ");
  var intensity = new Array(tokens.length);
  var max_intensity = Number.MIN_SAFE_INTEGER;
  var min_intensity = Number.MAX_SAFE_INTEGER;
  for (var i = 0; i < intensity.length; i++) {

      intensity[i] = wts[i];

      if (intensity[i] > max_intensity) {
          max_intensity = intensity[i];
      }
      if (intensity[i] < min_intensity) {
          min_intensity = intensity[i];
      }
  }
  var denominator = max_intensity - min_intensity;
  for (var i = 0; i < intensity.length; i++) {
  intensity[i] = (intensity[i] - min_intensity) / denominator;
  }

  var heat_text = "<p>";
  var space = "";
  for (var i = 0; i < tokens.length; i++) {
    if(useColor) {
      heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + "); font-size: " + (intensity[i]+0.5)*2 + "em'>" + space + tokens[i] + "</span>";
    } else {
      heat_text += "<span style='font-size: " + (intensity[i]+0.5)*2 + "em'>" + space + tokens[i] + "</span>";
    }
    if(useLines) {
      heat_text += "<br>"
    }
    if (space == "") {
    space = " ";
    }
  }
  $('#predictDiv').html(heat_text);
  $('#predictDiv').removeClass('hidden');
}

$("#predict").click(function(event){
    event.preventDefault();
    console.log("Predicting...")
    val = $("#text").val()
    $.ajax({
    url: "/api/bert",
    type: "POST",
    data: {
      text: val,
    },
    success: function(data, status){
      console.log("Data: " + JSON.stringify(data) + "\nStatus: " + status);
      wts = JSON.parse(data.result);
      text = data.text;
      show_predictions(text, wts);
    },
    error: function(error) {
      console.warn(error);
    },
  });
});